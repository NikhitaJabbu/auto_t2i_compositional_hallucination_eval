# object detection pipeline — gdino targeted + broad scan, sam coverage filter, countgd counting
import os
import json
import glob
import logging
import warnings
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    jsonl_path: str = "Outputs/parsed_prompts.jsonl"
    images_dir: str = "Outputs/Images_generated/"
    output_dir: str = "Outputs/detection_results/"
    weights_dir: str = "weights/"
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_model_type: str = "vit_h"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_download_url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    targeted_threshold: float = 0.20
    broad_threshold: float = 0.30
    match_confidence: float = 0.25
    extra_confidence: float = 0.50
    retry_threshold: float = 0.15
    mask_coverage_threshold: float = 0.30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    nms_iou_threshold: float = 0.35
    broad_nms_iou_threshold: float = 0.55  # looser than targeted to keep nearby same-class objects
    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".webp")
    save_crops: bool = True
    save_annotated_images: bool = True
    use_masked_crop: bool = False             # raw bbox crop — keeps natural colors for attribute detection
    countgd_repo_path: str = "CountGD"
    countgd_checkpoint: str = "weights/countgd_fsc147_best.pth"
    countgd_config: str = "config/cfg_fsc147_test.py"
    arrangement_alignment_tol: float = 0.08
    arrangement_cluster_dist: float = 0.25
    arrangement_grid_min_rc: int = 2

    @property
    def sam_weights_path(self):
        return os.path.join(self.weights_dir, self.sam_checkpoint)


STOP_WORDS = {"the", "a", "an", "of", "on", "in", "at", "is", "was", "and", "or"}
DUPLICATE_IOU_THRESHOLD    = 0.40
BOUNDARY_OVERLAP_THRESHOLD = 0.69  # both boxes must overlap each other >69% to flag collision


def compute_iou(box1, box2):
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return 0.0 if union <= 0 else inter / union


def compute_mutual_overlap(box1, box2):
    # returns (cov1, cov2) — how much of each box is covered by the other
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area1 = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    area2 = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    cov1 = inter / area1 if area1 > 0 else 0.0
    cov2 = inter / area2 if area2 > 0 else 0.0
    return cov1, cov2


@dataclass
class DetectedObject:
    label: str
    surface_form: str
    confidence: float
    bbox: list
    category: str                      # "correct" | "extra"
    obj_id: Optional[str] = None
    mask_area: Optional[int] = None
    mask_coverage: Optional[float] = None
    crop_path: Optional[str] = None
    annotated_image_path: Optional[str] = None


@dataclass
class MissingObject:
    obj_id: str
    label: str
    surface_form: str


@dataclass
class CountResult:
    label: str
    stage1_count: int
    countgd_count: Optional[int]
    final_count: int
    arrangement: str
    points: list = field(default_factory=list)
    bboxes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionReport:
    prompt_id: str
    prompt_text: str
    image_path: str
    objects_detected: list = field(default_factory=list)
    correct: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    counting_results: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    bbox_collision: bool = False
    colliding_pairs: list = field(default_factory=list)
    collision_dropped: list = field(default_factory=list)  # one dropped label per collision pair

    def to_dict(self):
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt_text,
            "image_path": self.image_path,
            "objects_detected": self.objects_detected,
            "correct": self.correct,
            "missing": self.missing,
            "extra": self.extra,
            "counting_results": self.counting_results,
            "bbox_collision": self.bbox_collision,
            "colliding_pairs": self.colliding_pairs,
            "collision_dropped": self.collision_dropped,
            "summary": self.summary,
        }


# lazy-loads gdino and sam on first use
class ModelLoader:
    def __init__(self, config):
        self.cfg = config
        self._gdino = None
        self._processor = None
        self._sam = None

    def _download_sam(self):
        path = self.cfg.sam_weights_path
        if os.path.isfile(path):
            return path
        os.makedirs(self.cfg.weights_dir, exist_ok=True)
        print("Downloading SAM weights...")
        urllib.request.urlretrieve(self.cfg.sam_download_url, path)
        return path

    @property
    def gdino(self):
        if self._gdino is None:
            print("Loading GroundingDINO...")
            self._processor = AutoProcessor.from_pretrained(self.cfg.gdino_model_id)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.cfg.gdino_model_id
            ).to(self.cfg.device)
            self._gdino.eval()
        return self._gdino

    @property
    def processor(self):
        if self._processor is None:
            _ = self.gdino
        return self._processor

    @property
    def sam(self):
        if self._sam is None:
            wp = self._download_sam()
            print("Loading SAM...")
            sam = sam_model_registry[self.cfg.sam_model_type](checkpoint=wp)
            sam.to(self.cfg.device)
            self._sam = SamPredictor(sam)
        return self._sam


def normalize_name(name):
    return " ".join(name.lower().strip().split())


def _stem(word):
    # strip simple plurals: chairs→chair, bananas→banana
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(phrase):
    return {_stem(w) for w in normalize_name(phrase).split()} - STOP_WORDS


def apply_nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    keep = torchvision.ops.nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_thresh,
    )
    return keep.tolist()


def load_image_for_sam(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gdino_query(model, processor, image, text_prompt, threshold, device):
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    labels = results.get("text_labels", results.get("labels", []))
    return boxes, scores, [normalize_name(str(l)) for l in labels]


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def safe_name(text):
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip().lower())
    out = "_".join(out.split("_"))
    return out[:80] if out else "obj"


def merge_bboxes(bboxes):
    # returns the union bbox covering all input boxes
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


# lazy-loads CountGD for open-vocab object counting, falls back to bbox count if unavailable
class _CountGDWrapper:

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._model = None
        self._transform = None
        self._args = None
        self._ready: Optional[bool] = None

    def _try_load(self) -> bool:
        import sys
        repo = self.cfg.countgd_repo_path
        ckpt = self.cfg.countgd_checkpoint
        if not os.path.isdir(repo):
            logger.warning("CountGD repo not found at '%s' — bbox-count fallback active.", repo)
            return False
        if not os.path.isfile(ckpt):
            logger.warning("CountGD checkpoint missing at '%s' — bbox-count fallback active.", ckpt)
            return False
        try:
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from main_inference import build_model_and_transforms
            from util.slconfig import SLConfig
            args = SLConfig.fromfile(os.path.join(repo, self.cfg.countgd_config))
            args.pretrain_model_path = ckpt
            args.device = self.cfg.device
            model, transform = build_model_and_transforms(args)
            model.eval()
            self._model     = model
            self._transform = transform
            self._args      = args
            logger.info("CountGD loaded.")
            return True
        except Exception as exc:
            logger.warning("CountGD load failed (%s) — bbox-count fallback active.", exc)
            return False

    @property
    def ready(self) -> bool:
        if self._ready is None:
            self._ready = self._try_load()
        return self._ready

    def count(self, image: Image.Image, text: str):
        # returns (count, point_list) or (None, []) if unavailable
        if not self.ready:
            return None, []
        try:
            import torch
            from main_inference import run_inference_single
            with torch.no_grad():
                result = run_inference_single(
                    self._model, self._transform, image, text + ".", self._args
                )
            pts = result.get("points", [])
            return int(result.get("count", 0)), [(float(p[0]), float(p[1])) for p in pts]
        except Exception as exc:
            logger.warning("CountGD inference error for '%s': %s", text, exc)
            return None, []


# tags spatial arrangement of detected objects from their centroids: row/column/grid/cluster/scattered
class _ArrangementTagger:

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg

    def tag(self, centroids: list, img_w: int, img_h: int) -> str:
        n = len(centroids)
        if n == 0: return "none"
        if n == 1: return "single"
        if n == 2: return "pair"
        pts = np.array(centroids, dtype=np.float64)
        pts[:, 0] /= max(img_w, 1)
        pts[:, 1] /= max(img_h, 1)
        xs, ys = pts[:, 0], pts[:, 1]
        tol = self.cfg.arrangement_alignment_tol
        if np.std(ys) < tol and np.std(xs) > tol:
            return "row"
        if np.std(xs) < tol and np.std(ys) > tol:
            return "column"
        min_rc = self.cfg.arrangement_grid_min_rc
        if self._n_groups(xs, tol) >= min_rc and self._n_groups(ys, tol) >= min_rc:
            return "grid"
        if self._avg_pair_dist(pts) < self.cfg.arrangement_cluster_dist:
            return "cluster"
        return "scattered"

    @staticmethod
    def _n_groups(values: np.ndarray, tol: float) -> int:
        sv = np.sort(values)
        return 1 + int(np.sum(np.diff(sv) > tol))

    @staticmethod
    def _avg_pair_dist(pts: np.ndarray) -> float:
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        n = len(pts)
        return float(dists[np.triu_indices(n, k=1)].mean())


class ObjectDetectionPipeline:
    def __init__(self, config=None):
        self.cfg    = config or DetectionConfig()
        self.models = ModelLoader(self.cfg)
        self._countgd = _CountGDWrapper(self.cfg)
        self._tagger  = _ArrangementTagger(self.cfg)

    def detect(self, image, phrases, threshold=None, objects=None, count_obj_ids=None):
        # targeted gdino scan (surface + bare noun per object) + broad scan with dynamic prompt
        if threshold is None:
            threshold = self.cfg.targeted_threshold
        count_obj_ids = count_obj_ids or set()

        # count objects use bare noun only; normal objects query surface form + bare noun
        query_phrases = []
        if objects:
            seen_queries: set = set()
            for obj in objects:
                bare    = normalize_name(obj["name"])
                surface = normalize_name(obj.get("surface", obj["name"]))
                if obj.get("id") in count_obj_ids:
                    if bare not in seen_queries:
                        query_phrases.append(bare)
                        seen_queries.add(bare)
                else:
                    if surface not in seen_queries:
                        query_phrases.append(surface)
                        seen_queries.add(surface)
                    if bare not in seen_queries and bare != surface:
                        query_phrases.append(bare)
                        seen_queries.add(bare)
        else:
            query_phrases = list(phrases)

        targeted_boxes, targeted_scores, targeted_labels = [], [], []
        for p in query_phrases:
            b, s, l = gdino_query(
                self.models.gdino, self.models.processor,
                image, f"{p}.", threshold, self.cfg.device,
            )
            targeted_boxes += b; targeted_scores += s; targeted_labels += l

        if targeted_boxes:
            keep = apply_nms(targeted_boxes, targeted_scores, self.cfg.nms_iou_threshold)
            targeted_boxes  = [targeted_boxes[i]  for i in keep]
            targeted_scores = [targeted_scores[i] for i in keep]
            targeted_labels = [targeted_labels[i] for i in keep]

        # broad prompt = base categories + any expected object names not in base
        _broad_base = {
            "person", "animal", "vehicle", "chair", "table", "sofa", "bed", "shelf",
            "plant", "tree", "flower", "food", "fruit", "bowl", "bottle", "cup", "bag",
            "phone", "laptop", "tv", "clock", "lamp", "mirror", "frame", "toy", "book",
        }
        extra_terms = [
            normalize_name(obj["name"]) for obj in (objects or [])
            if normalize_name(obj["name"]) not in _broad_base
        ]
        broad_prompt = ". ".join(sorted(_broad_base) + extra_terms) + "."

        broad_boxes, broad_scores, broad_labels = gdino_query(
            self.models.gdino, self.models.processor,
            image, broad_prompt, self.cfg.broad_threshold, self.cfg.device,
        )
        if broad_boxes:
            keep = apply_nms(broad_boxes, broad_scores, self.cfg.broad_nms_iou_threshold)
            broad_boxes  = [broad_boxes[i]  for i in keep]
            broad_scores = [broad_scores[i] for i in keep]
            broad_labels = [broad_labels[i] for i in keep]

        return (
            (targeted_boxes, targeted_scores, targeted_labels),
            (broad_boxes,    broad_scores,    broad_labels),
        )

    # ----------------------------------------------------------
    # SEGMENT
    # ----------------------------------------------------------
    def segment(self, image_path, boxes):
        image = load_image_for_sam(image_path)
        self.models.sam.set_image(image)
        masks, areas = [], []
        for box in boxes:
            masks_pred, scores, _ = self.models.sam.predict(
                box=np.array(box), multimask_output=True
            )
            best_idx  = int(np.argmax(scores))
            best_mask = masks_pred[best_idx]
            masks.append(best_mask)
            areas.append(int(best_mask.sum()))
        return masks, areas

    # ----------------------------------------------------------
    # MATCH HELPERS
    def _find_best_match(self, label_clean, expected_names, matched_expected):
        # exact/substring match first, then best content-word overlap
        label_words = content_words(label_clean)

        for exp in expected_names:           # pass 1: exact / substring
            if exp in matched_expected:
                continue
            if label_clean == exp or label_clean in exp or exp in label_clean:
                return exp

        best_exp, best_overlap = None, 0    # pass 2: content-word overlap
        for exp in expected_names:
            if exp in matched_expected:
                continue
            overlap = len(label_words & content_words(exp))
            if overlap > best_overlap:
                best_overlap = overlap
                best_exp = exp

        return best_exp if best_overlap > 0 else None

    def match(self, objects, detections, broad_indices):
        # maps detections to expected objects; splits into correct/missing/extra
        boxes, scores, labels, mask_areas = detections

        expected_map = {
            normalize_name(obj.get("surface", obj["name"])): obj
            for obj in objects
        }
        expected_names  = set(expected_map.keys())
        detections_list = list(zip(boxes, scores, labels, mask_areas))

        correct, missing, extra = [], [], []
        matched_expected, matched_det = set(), set()

        # ---------- correct ----------
        for det_idx, (box, score, label, m_area) in sorted(
            enumerate(detections_list), key=lambda x: x[1][1], reverse=True
        ):
            if score < self.cfg.match_confidence:
                continue
            label_clean = normalize_name(label)
            best_match  = self._find_best_match(label_clean, expected_names, matched_expected)
            if best_match:
                obj       = expected_map[best_match]
                bbox_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
                correct.append(DetectedObject(
                    label=obj["name"],
                    surface_form=obj.get("surface", obj["name"]),
                    confidence=round(score, 4),
                    bbox=[round(v, 2) for v in box],
                    category="correct",
                    obj_id=obj["id"],
                    mask_area=m_area,
                    mask_coverage=round(m_area / bbox_area, 4),
                ))
                matched_expected.add(best_match)
                matched_det.add(det_idx)

        # ---------- missing ----------
        for exp, obj in expected_map.items():
            if exp not in matched_expected:
                missing.append(MissingObject(
                    obj_id=obj["id"],
                    label=obj["name"],
                    surface_form=obj.get("surface", obj["name"]),
                ))

        GENERIC_LABELS = {"object", "thing", "item", "sign"}
        extra_seen = set()

        for det_idx, (box, score, label, m_area) in enumerate(detections_list):
            if det_idx in matched_det:
                continue
            label_clean = normalize_name(label)
            label_words = frozenset(label_clean.split()) - STOP_WORDS
            if len(label_words) < 1:
                continue
            if label_clean in GENERIC_LABELS:
                continue
            threshold = self.cfg.extra_confidence if det_idx in broad_indices \
                        else self.cfg.match_confidence
            if score < threshold:
                continue
            label_fp = tuple(sorted(label_words))
            if label_fp in extra_seen:
                continue
            suppress = any(
                (label_words & (content_words(c.surface_form) | content_words(c.label)))
                and compute_iou(box, c.bbox) >= DUPLICATE_IOU_THRESHOLD
                for c in correct
            )
            if suppress:
                continue
            extra_seen.add(label_fp)
            bbox_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
            extra.append(DetectedObject(
                label=label_clean,
                surface_form=label_clean,
                confidence=round(score, 4),
                bbox=[round(v, 2) for v in box],
                category="extra",
                mask_area=m_area,
                mask_coverage=round(m_area / bbox_area, 4),
            ))

        return correct, missing, extra

    def retry_missing(self, image, image_path, missing_objs):
        # re-queries gdino at lower threshold for objects not found in main scan
        if not missing_objs:
            return [], missing_objs

        phrases = list({normalize_name(o.surface_form) for o in missing_objs})
        logger.info("Retry pass for %d missing: %s", len(phrases), phrases)

        retry_boxes, retry_scores, retry_labels = [], [], []
        for p in phrases:
            b, s, l = gdino_query(
                self.models.gdino, self.models.processor,
                image, f"{p}.", self.cfg.retry_threshold, self.cfg.device,
            )
            retry_boxes += b; retry_scores += s; retry_labels += l

        if not retry_boxes:
            return [], missing_objs

        if len(retry_boxes) > 1:
            keep = apply_nms(retry_boxes, retry_scores, self.cfg.nms_iou_threshold)
            retry_boxes  = [retry_boxes[i]  for i in keep]
            retry_scores = [retry_scores[i] for i in keep]
            retry_labels = [retry_labels[i] for i in keep]

        _, retry_areas = self.segment(image_path, retry_boxes)

        expected_map     = {normalize_name(o.surface_form): o for o in missing_objs}
        expected_names   = set(expected_map.keys())
        matched_expected = set()
        recovered        = []

        for box, score, label, m_area in sorted(
            zip(retry_boxes, retry_scores, retry_labels, retry_areas),
            key=lambda x: x[1], reverse=True,
        ):
            if score < self.cfg.match_confidence:
                continue
            label_clean = normalize_name(label)
            best_match  = self._find_best_match(label_clean, expected_names, matched_expected)
            if best_match:
                mo        = expected_map[best_match]
                bbox_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
                cov       = m_area / bbox_area
                if cov < self.cfg.mask_coverage_threshold:
                    continue
                recovered.append(DetectedObject(
                    label=mo.label,
                    surface_form=mo.surface_form,
                    confidence=round(score, 4),
                    bbox=[round(v, 2) for v in box],
                    category="correct",
                    obj_id=mo.obj_id,
                    mask_area=m_area,
                    mask_coverage=round(cov, 4),
                ))
                matched_expected.add(best_match)
                logger.info("Retry recovered: '%s' (%.4f)", mo.label, score)

        still_missing = [
            o for o in missing_objs
            if normalize_name(o.surface_form) not in matched_expected
        ]
        return recovered, still_missing

    def run_counting(self, image_path: str, detected_objects: list,
                     img_w: int, img_h: int) -> list:
        # groups detections by label, runs countgd per group, tags arrangement from centroids
        image = Image.open(image_path).convert("RGB")

        canonical_map = []  # list of [canonical_label, words, [objs]]
        for obj in detected_objects:
            obj_words = content_words(obj.label)
            merged = False
            for entry in canonical_map:
                if obj_words <= entry[1] or entry[1] <= obj_words:
                    if len(obj_words) < len(entry[1]):
                        entry[1] = obj_words
                        entry[0] = " ".join(
                            w for w in normalize_name(obj.label).split()
                            if w not in STOP_WORDS
                        )
                    entry[2].append(obj)
                    merged = True
                    break
            if not merged:
                canonical_map.append([
                    " ".join(w for w in normalize_name(obj.label).split() if w not in STOP_WORDS),
                    obj_words,
                    [obj],
                ])

        groups = {entry[0]: entry[2] for entry in canonical_map}
        results = []

        for label, objs in groups.items():
            bboxes       = [o.bbox for o in objs]
            stage1_count = len(objs)
            centroids    = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bboxes]

            countgd_count, countgd_pts = self._countgd.count(image, label)

            if countgd_count is not None:
                final_count     = countgd_count
                arrangement_pts = countgd_pts if countgd_pts else centroids
            else:
                final_count     = stage1_count
                arrangement_pts = centroids

            arrangement = self._tagger.tag(arrangement_pts, img_w, img_h)

            logger.info(
                "Stage2 [%s]: stage1=%d  countgd=%s  final=%d  arrangement=%s",
                label, stage1_count,
                countgd_count if countgd_count is not None else "N/A",
                final_count, arrangement,
            )
            results.append(CountResult(
                label=label,
                stage1_count=stage1_count,
                countgd_count=countgd_count,
                final_count=final_count,
                arrangement=arrangement,
                points=[(round(p[0], 2), round(p[1], 2)) for p in arrangement_pts],
                bboxes=bboxes,
            ))

        return results

    def save_detection_artifacts(self, image_path, prompt_id,
                                  all_objects, correct_objects,
                                  masks_all, boxes_all, labels_all):
        # saves annotated image (correct=green, extra=red) and one crop per correct label
        full_rgb = load_image_for_sam(image_path)
        h, w     = full_rgb.shape[:2]

        prompt_crop_dir = os.path.join(self.cfg.output_dir, "crops", prompt_id)
        os.makedirs(prompt_crop_dir, exist_ok=True)
        annotated_path = None

        # --- Annotated image: draw every detection (correct=green, extra=red) ---
        if self.cfg.save_annotated_images:
            canvas = cv2.cvtColor(full_rgb.copy(), cv2.COLOR_RGB2BGR)
            for obj in all_objects:
                x1, y1, x2, y2 = clamp_box(obj.bbox, w, h)
                color = (0, 255, 0) if obj.category == "correct" else (0, 0, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    canvas, f"{obj.label} {obj.confidence:.2f}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                )
            annotated_dir  = os.path.join(self.cfg.output_dir, "annotated")
            os.makedirs(annotated_dir, exist_ok=True)
            annotated_path = os.path.join(annotated_dir, f"{prompt_id}_annotated.jpg")
            cv2.imwrite(annotated_path, canvas)

        # --- Crops: one per unique label (highest confidence), correct only ---
        label_groups = {}
        for obj in correct_objects:
            key = safe_name(obj.label)
            label_groups.setdefault(key, []).append(obj)

        for label_key, group in label_groups.items():
            best_obj = max(group, key=lambda o: o.confidence)
            x1, y1, x2, y2 = clamp_box(best_obj.bbox, w, h)
            crop_path = os.path.join(prompt_crop_dir, f"{label_key}.png")

            matched_mask, best_iou = None, 0.0
            for box_ref, mask_ref, _ in zip(boxes_all, masks_all, labels_all):
                iou = compute_iou(best_obj.bbox, box_ref)
                if iou > best_iou:
                    best_iou     = iou
                    matched_mask = mask_ref

            # Use SAM mask tight bbox when available — gives a clean crop of just
            # the object surface without surrounding objects or black background.
            # Falls back to detection bbox when no mask is available.
            if matched_mask is not None:
                mask_ys, mask_xs = np.where(matched_mask)
                if len(mask_xs) > 0:
                    mx1 = max(0, int(mask_xs.min()))
                    my1 = max(0, int(mask_ys.min()))
                    mx2 = min(w, int(mask_xs.max()) + 1)
                    my2 = min(h, int(mask_ys.max()) + 1)
                    crop_rgb = full_rgb[my1:my2, mx1:mx2].copy()
                    # Update bbox to mask-tight coords so attributeprediction
                    # uses the correct region for overlap calculations
                    best_obj.bbox = [float(mx1), float(my1), float(mx2), float(my2)]
                else:
                    crop_rgb = full_rgb[y1:y2, x1:x2].copy()
            else:
                crop_rgb = full_rgb[y1:y2, x1:x2].copy()

            if crop_rgb.size == 0:
                logger.warning("Empty crop skipped for %s", best_obj.label)
                continue

            cv2.imwrite(crop_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
            best_obj.crop_path = crop_path

        if annotated_path:
            for obj in correct_objects:
                obj.annotated_image_path = annotated_path

    # ----------------------------------------------------------
    def run_single(self, image_path, prompt_data):
        # runs full detection pipeline for one prompt and returns a DetectionReport
        pid         = prompt_data["id"]
        prompt_text = prompt_data.get("prompt", "")
        objects     = prompt_data["objects"]

        image        = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        attributes    = prompt_data.get("attributes", [])
        count_obj_ids = {a["obj"] for a in attributes if a.get("type") == "count"}

        (t_boxes, t_scores, t_labels), (b_boxes, b_scores, b_labels) = \
            self.detect(image, [], objects=objects, count_obj_ids=count_obj_ids)

        boxes  = t_boxes  + b_boxes
        scores = t_scores + b_scores
        labels = t_labels + b_labels
        broad_indices = set(range(len(t_boxes), len(boxes)))

        masks, areas = self.segment(image_path, boxes)

        # Coverage gate applied to ALL detections — targeted and broad alike.
        # Real objects almost always have SAM coverage ≥ 0.30;
        # false-positive boxes over flat backgrounds/walls do not.
        filtered = []
        for i, (box, score, label, area, mask) in \
                enumerate(zip(boxes, scores, labels, areas, masks)):
            is_broad  = i in broad_indices
            bbox_area = max((box[2] - box[0]) * (box[3] - box[1]), 1)
            if area / bbox_area < self.cfg.mask_coverage_threshold:
                continue
            filtered.append((box, score, label, area, mask, is_broad))

        if filtered:
            boxes, scores, labels, areas, masks, is_broad = zip(*filtered)
            boxes, scores, labels, areas, masks = \
                list(boxes), list(scores), list(labels), list(areas), list(masks)
            broad_indices = {i for i, flag in enumerate(is_broad) if flag}
        else:
            boxes, scores, labels, areas, masks, broad_indices = [], [], [], [], [], set()

        correct, missing, extra = self.match(
            objects, (boxes, scores, labels, areas), broad_indices
        )

        # retry pass
        if missing:
            recovered, missing = self.retry_missing(image, image_path, missing)
            correct.extend(recovered)
            for obj in recovered:
                boxes.append(obj.bbox)
                scores.append(obj.confidence)
                labels.append(obj.surface_form)

        all_kept = correct + extra
        if self.cfg.save_crops and all_kept:
            self.save_detection_artifacts(
                image_path=image_path,
                prompt_id=pid,
                all_objects=all_kept,
                correct_objects=correct,
                masks_all=masks,
                boxes_all=boxes[:len(masks)],
                labels_all=labels[:len(masks)],
            )

        counting_results = self.run_counting(image_path, all_kept, img_w, img_h)

        # deduplicate detections by label, merging bboxes for same-label entries
        seen_entries = []
        for b, s, l in zip(boxes, scores, labels):
            l_norm      = normalize_name(l)
            l_words     = content_words(l_norm)
            if not l_words:
                continue
            rounded_box = [round(v, 2) for v in b]
            merged = False
            for entry in seen_entries:
                if l_words <= entry["words"] or entry["words"] <= l_words:
                    if len(l_words) < len(entry["words"]):
                        entry["label"] = " ".join(
                            w for w in l_norm.split() if w not in STOP_WORDS
                        )
                        entry["words"] = l_words
                    entry["score"] = max(entry["score"], round(float(s), 4))
                    entry["count"] += 1
                    entry["bboxes"].append(rounded_box)
                    merged = True
                    break
            if not merged:
                seen_entries.append({
                    "label":  " ".join(w for w in l_norm.split() if w not in STOP_WORDS),
                    "score":  round(float(s), 4),
                    "words":  l_words,
                    "count":  1,
                    "bboxes": [rounded_box],              # start bbox list
                })

        objects_detected = [
            {
                "label": e["label"],
                "score": e["score"],
                "count": e["count"],
                "bbox":  e["bboxes"][0] if e["count"] == 1 else merge_bboxes(e["bboxes"]),
            }
            for e in seen_entries
        ]

        expected_names = {normalize_name(o.get("surface", o["name"])) for o in objects}
        expected_words = {name: content_words(name) for name in expected_names}

        exp_base_names = {}
        for obj in objects:
            exp_key = normalize_name(obj.get("surface", obj["name"]))
            exp_base_names[exp_key] = obj["name"]

        correct_out     = []
        extra_out       = []
        matched_exp_out = set()

        for entry in seen_entries:
            det_label = entry["label"]
            det_words = content_words(det_label)

            # best word-overlap match prevents shared adjectives (e.g. "black") hijacking the wrong object
            best_exp, best_overlap = None, 0
            for exp_name, exp_words in expected_words.items():
                if exp_name in matched_exp_out:
                    continue
                overlap = len(det_words & exp_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_exp     = exp_name

            bbox = entry["bboxes"][0] if entry["count"] == 1 else merge_bboxes(entry["bboxes"])
            if best_exp is not None and best_overlap > 0:
                matched_name = exp_base_names.get(best_exp, det_label)
                matched_exp_out.add(best_exp)
                correct_out.append({
                    "label":      matched_name,
                    "bbox":       bbox,
                    "confidence": entry["score"],
                })
            else:
                extra_out.append({
                    "label": det_label,
                    "bbox":  bbox,
                })

        # missing = expected objects not matched by any detection
        matched_expected = set()
        for exp_name, exp_words in expected_words.items():
            for entry in seen_entries:
                if content_words(entry["label"]) & exp_words:
                    matched_expected.add(exp_name)
                    break

        missing_out = []
        for obj in objects:
            exp_key = normalize_name(obj.get("surface", obj["name"]))
            if exp_key not in matched_expected:
                missing_out.append({"label": obj["name"]})

        simple_counting = [
            {
                "label": r.label,
                "count": r.final_count,
                "bbox":  r.bboxes[0] if len(r.bboxes) == 1 else merge_bboxes(r.bboxes),
            }
            for r in counting_results
        ]

        # flag collisions — two correct detections mapped to the same image region
        colliding_pairs = []
        for i in range(len(correct_out)):
            for j in range(i + 1, len(correct_out)):
                box_a = correct_out[i]["bbox"]
                box_b = correct_out[j]["bbox"]
                cov_a, cov_b = compute_mutual_overlap(box_a, box_b)

                collision_reason = None
                if cov_a > BOUNDARY_OVERLAP_THRESHOLD and cov_b > BOUNDARY_OVERLAP_THRESHOLD:
                    collision_reason = (
                        f"mutual boundary overlap — "
                        f"'{correct_out[i]['label']}' covers {cov_a*100:.1f}% of "
                        f"'{correct_out[j]['label']}' and vice versa {cov_b*100:.1f}%"
                    )

                if collision_reason:
                    colliding_pairs.append({
                        "obj_a":            correct_out[i]["label"],
                        "obj_b":            correct_out[j]["label"],
                        "bbox_a":           box_a,
                        "bbox_b":           box_b,
                        "cov_a":            round(cov_a, 4),
                        "cov_b":            round(cov_b, 4),
                        "collision_reason": collision_reason,
                    })
                    logger.info(
                        "BBox collision in prompt '%s': '%s' vs '%s' — %s",
                        pid, correct_out[i]["label"], correct_out[j]["label"], collision_reason,
                    )
        bbox_collision = len(colliding_pairs) > 0

        # For each collision pair keep the higher-confidence detection, drop the lower.
        # Greedy pass handles chains (A-B, B-C).
        conf_lookup = {c["label"]: c["confidence"] for c in correct_out}
        dropped_set: set = set()
        for pair in colliding_pairs:
            a_lbl, b_lbl = pair["obj_a"], pair["obj_b"]
            if a_lbl in dropped_set or b_lbl in dropped_set:
                continue  # one side already resolved
            conf_a = conf_lookup.get(a_lbl, 0.0)
            conf_b = conf_lookup.get(b_lbl, 0.0)
            dropped_set.add(b_lbl if conf_a >= conf_b else a_lbl)
        collision_dropped = [{"label": lbl} for lbl in dropped_set]
        correct_final = [c for c in correct_out if c["label"] not in dropped_set]
        missing_out  += [{"label": lbl} for lbl in dropped_set]

        return DetectionReport(
            prompt_id=pid,
            prompt_text=prompt_text,
            image_path=image_path,
            objects_detected=objects_detected,
            correct=correct_final,
            missing=missing_out,
            extra=extra_out,
            counting_results=simple_counting,
            bbox_collision=bbox_collision,
            colliding_pairs=colliding_pairs,
            collision_dropped=collision_dropped,
            summary={
                "total_expected":  len(objects),
                "total_detected":  len(objects_detected),
                "total_correct":   len(correct_final),
                "total_missing":   len(missing_out),
                "total_extra":     len(extra_out),
                "bbox_collision":  bbox_collision,
            },
        )

    # ----------------------------------------------------------
    # RUN BATCH
    # ----------------------------------------------------------
    def run_batch(self, jsonl_path, images_dir, output_dir):
        # runs detection on all prompts, saves results, prints hallucination scores
        os.makedirs(output_dir, exist_ok=True)

        prompts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))

        reports, skipped, failed = [], [], []
        collision_exclusions     = {}   # prompt_id -> [(obj_a, obj_b), ...]

        for pd in prompts:
            pid      = pd["id"]
            img_path = None
            for ext in self.cfg.image_extensions:
                exact = os.path.join(images_dir, f"{pid}{ext}")
                if os.path.isfile(exact):
                    img_path = exact
                    break
                matches = sorted(glob.glob(os.path.join(images_dir, f"{pid}_*{ext}")))
                if matches:
                    img_path = matches[0]
                    break

            if img_path is None:
                skipped.append(pid)
                continue

            try:
                rep = self.run_single(img_path, pd)
                rep_dict = rep.to_dict()
                reports.append(rep_dict)
                if rep.bbox_collision:
                    collision_exclusions[pid] = [d["label"] for d in rep.collision_dropped]
            except Exception as e:
                logger.error("Failed %s: %s", pid, e)
                failed.append({"id": pid, "error": str(e)})

        with open(os.path.join(output_dir, "detection_results.json"), "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)

        tc = sum(r["summary"]["total_correct"]  for r in reports)
        tm = sum(r["summary"]["total_missing"]  for r in reports)
        te = sum(r["summary"]["total_extra"]    for r in reports)
        td = sum(r["summary"]["total_detected"] for r in reports)
        tt = sum(r["summary"]["total_expected"] for r in reports)

        n_colliding_prompts = len(collision_exclusions)
        n_collision_pairs   = sum(len(r.get("colliding_pairs", [])) for r in reports)
        n_coll              = sum(len(dropped) for dropped in collision_exclusions.values())

        print(f"\n{'='*60}")
        print(f"  Batch complete")
        print(f"  Images processed      : {len(reports)}")
        print(f"  Images skipped        : {len(skipped)}")
        print(f"  Images failed         : {len(failed)}")
        print(f"  Expected objects      : {tt}")
        print(f"  Correct               : {tc}")
        print(f"  Missing               : {tm}")
        print(f"  Extra                 : {te}")
        if collision_exclusions:
            print(f"  Collision pairs       : {n_collision_pairs} across {n_colliding_prompts} prompt(s)")
        print(f"{'='*60}")

        go  = tt
        co  = tc
        eo  = te
        to_ = td
        missed_hallucination = (go - co) / go if go > 0 else 0.0
        extra_hallucination  = eo / to_       if to_ > 0 else 0.0
        print(f"\n  Object Hallucination (missed) : {missed_hallucination:.4f}  [ (Go-Co)/Go = ({go}-{co})/{go} ]")
        print(f"  Object Hallucination (extra)  : {extra_hallucination:.4f}  [ Eo/To = {eo}/{to_} ]")

        det_summary = {
            "H_obj_missed": round(missed_hallucination, 4),
            "H_obj_extra":  round(extra_hallucination,  4),
            "Go": go, "Co": co, "Eo": eo, "To": to_,
        }
        with open(os.path.join(output_dir, "detection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(det_summary, f, indent=2)

        return reports, collision_exclusions


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    cfg = DetectionConfig()
    print("\nObject Detection Pipeline v5 + Stage 2 Counting")
    print("Device:", cfg.device)
    pipeline = ObjectDetectionPipeline(cfg)
    reports, collision_exclusions = \
        pipeline.run_batch(cfg.jsonl_path, cfg.images_dir, cfg.output_dir)