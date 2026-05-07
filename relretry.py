"""
relretry.py — predicts spatial relations between detected object pairs.

Uses bounding box geometry and SAM mask centroids.
Spatial vocab loaded from vocab/spatial.txt.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Set

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM
    sam_checkpoint: str = "weights/sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"

    # Padding around pair union bbox when cropping
    crop_pad: float = 0.05

    # Spatial geometry thresholds
    containment_threshold: float = 0.70
    bg_containment_threshold: float = 0.40
    bg_area_ratio: float = 0.60
    bg_span_ratio: float = 0.75
    direction_ratio: float = 1.5
    near_threshold: float = 0.30
    on_gap_ratio: float = 0.08
    on_overlap_ratio: float = 0.45
    on_contact_ratio: float = 1.5
    on_min_containment: float = 0.05
    both_large_threshold: float = 0.15

    # I/O paths
    detection_json:      str = "Outputs/detection_results/detection_results.json"
    attribute_json:      str = "Outputs/detection_results/attribute_results.json"
    parsed_prompts_json: str = "Outputs/parsed_prompts.jsonl"
    output_dir:          str = "Outputs/detection_results"
    vocab_dir:           str = "vocab"


# ---------------------------------------------------------------------------
# Vocabulary loader
# ---------------------------------------------------------------------------

def load_vocab(vocab_path: str) -> Set[str]:
    """Load vocabulary from text file, one term per line."""
    vocab = set()
    try:
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip().lower()
                    if term and not term.startswith('#'):
                        vocab.add(term)
            logger.info(f"Loaded {len(vocab)} terms from {vocab_path}")
    except Exception as e:
        logger.warning(f"Could not load vocab from {vocab_path}: {e}")
    return vocab


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MaskResult:
    centroid: tuple
    mask: Optional[np.ndarray] = None




@dataclass
class PairRelationResult:
    """Final output for one object pair."""
    from_obj:   str
    to_obj:     str
    spatial:    Optional[str]
    confidence: float


# ---------------------------------------------------------------------------
# Mask centroid (SAM → GrabCut → bbox centre)
# ---------------------------------------------------------------------------

class MaskCentroid:
    def __init__(self, cfg: Config, sam_predictor=None):
        self.cfg = cfg
        if sam_predictor is not None:
            self._predictor = sam_predictor
            self._ready = True
            logger.info("SAM reused from detection pipeline.")
        else:
            self._predictor = None
            self._ready: Optional[bool] = None

    def _try_load(self) -> bool:
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[self.cfg.sam_model_type](
                checkpoint=self.cfg.sam_checkpoint)
            sam.to(self.cfg.device).eval()
            self._predictor = SamPredictor(sam)
            logger.info("SAM loaded.")
            return True
        except Exception as exc:
            logger.warning("SAM unavailable (%s).", exc)
            return False

    @property
    def ready(self) -> bool:
        if self._ready is None:
            self._ready = self._try_load()
        return self._ready

    def _sam(self, img_np: np.ndarray, bbox: list) -> Optional[MaskResult]:
        try:
            self._predictor.set_image(img_np)
            masks, scores, _ = self._predictor.predict(
                box=np.array(bbox, dtype=float), multimask_output=True)
            best = masks[int(np.argmax(scores))]
            rows, cols = np.where(best)
            if len(rows) < 5:
                return None
            return MaskResult(
                centroid=(float(np.mean(cols)), float(np.mean(rows))),
                mask=best,
            )
        except Exception as exc:
            logger.debug("SAM failed: %s", exc)
            return None

    @staticmethod
    def _grabcut(img_np: np.ndarray, bbox: list) -> Optional[MaskResult]:
        try:
            import cv2
            x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                return None
            gc = np.zeros(img_np.shape[:2], np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_np, gc, (x1, y1, x2 - x1, y2 - y1),
                        bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc == 2) | (gc == 0), 0, 1).astype(bool)
            rows, cols = np.where(fg)
            if len(rows) < 10:
                return None
            return MaskResult(
                centroid=(float(np.mean(cols)), float(np.mean(rows))),
                mask=fg,
            )
        except Exception as exc:
            logger.debug("GrabCut failed: %s", exc)
            return None

    def compute(self, img_np: np.ndarray, bbox: list, label: str = "") -> MaskResult:
        if self.ready:
            r = self._sam(img_np, bbox)
            if r:
                return r
        r = self._grabcut(img_np, bbox)
        if r:
            return r
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return MaskResult(centroid=(cx, cy), mask=None)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def crop_pair(img: Image.Image, box_a: list, box_b: list,
              pad: float = 0.05) -> tuple:
    """Return (crop, (ox, oy)) — tight union crop with padding."""
    w, h = img.size
    x1 = max(0, min(box_a[0], box_b[0]))
    y1 = max(0, min(box_a[1], box_b[1]))
    x2 = min(w, max(box_a[2], box_b[2]))
    y2 = min(h, max(box_a[3], box_b[3]))
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(w, int(x2 + pw))
    y2 = min(h, int(y2 + ph))
    return img.crop((x1, y1, x2, y2)), (x1, y1)



# ---------------------------------------------------------------------------
# Spatial geometry
# ---------------------------------------------------------------------------

class SpatialGeometry:
    """Computes spatial relations from geometry and external vocab."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._img_w = self._img_h = self._img_area = 1
        
        # Load spatial vocabulary
        spatial_vocab_path = os.path.join(cfg.vocab_dir, "spatial.txt")
        self.bg_surface_labels = load_vocab(spatial_vocab_path)

    @staticmethod
    def _containment(box_a, box_b):
        ix1 = max(box_a[0], box_b[0])
        ix2 = min(box_a[2], box_b[2])
        iy1 = max(box_a[1], box_b[1])
        iy2 = min(box_a[3], box_b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0, 0.0
        area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        return inter / area_a, inter / area_b

    def _is_bg(self, box, label: str = "") -> bool:
        lbl = label.lower().strip()
        if lbl in self.bg_surface_labels:
            return True
        w = box[2] - box[0]
        h = box[3] - box[1]
        return ((w * h / self._img_area) > self.cfg.bg_area_ratio and
                w / self._img_w > self.cfg.bg_span_ratio and
                h / self._img_h > self.cfg.bg_span_ratio)

    def compute(self, box_a, centroid_a, box_b, centroid_b,
                img_w, img_h, label_a: str = "", label_b: str = "",
                count_a: int = 1, count_b: int = 1) -> str:
        self._img_w = img_w
        self._img_h = img_h
        self._img_area = max(img_w * img_h, 1)

        multi = (count_a > 1 or count_b > 1)
        a_bg = self._is_bg(box_a, label_a)
        b_bg = self._is_bg(box_b, label_b)
        neither_bg = not a_bg and not b_bg
        cont_a, cont_b = self._containment(box_a, box_b)

        # Containment relations
        if neither_bg and not multi:
            thr = self.cfg.containment_threshold
            if cont_a >= thr and cont_b < thr:
                return "inside"
            if cont_b >= thr and cont_a < thr:
                return "contains"
        elif a_bg != b_bg and not multi:
            thr = self.cfg.bg_containment_threshold
            if not a_bg and cont_a >= thr:
                return "on"
            if not b_bg and cont_b >= thr:
                return "on"

        # Directional relations
        cx_a, cy_a = centroid_a
        cx_b, cy_b = centroid_b
        diag = math.sqrt(img_w ** 2 + img_h ** 2) or 1.0
        raw_dx = cx_b - cx_a
        raw_dy = cy_b - cy_a
        dist = math.sqrt(raw_dx ** 2 + raw_dy ** 2) / diag
        dx = raw_dx / (img_w or 1)
        dy = raw_dy / (img_h or 1)
        adx, ady = abs(dx), abs(dy)

        if dist < self.cfg.near_threshold * 0.5:
            return "next to"
        
        ratio = self.cfg.direction_ratio
        if adx > ratio * ady:
            return "left of" if dx > 0 else "right of"
        if ady > ratio * adx:
            return "above" if dy > 0 else "below"
        if dist < self.cfg.near_threshold:
            return "near"
        if adx >= ady:
            return "left of" if dx > 0 else "right of"
        return "above" if dy > 0 else "below"



# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

class RelationPredictor:
    """Main orchestrator for all relations."""

    def __init__(self, cfg: Config = None, sam_predictor=None):
        self.cfg = cfg or Config()
        
        # Print system info
        print("\n" + "=" * 60)
        print("Initializing RelationPredictor")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 60 + "\n")
        
        self._mc = MaskCentroid(self.cfg, sam_predictor=sam_predictor)
        self._geo = SpatialGeometry(self.cfg)

    def predict_pair(self, img: Image.Image, img_np: np.ndarray,
                     label_a: str, box_a: list,
                     label_b: str, box_b: list,
                     count_a: int = 1, count_b: int = 1) -> PairRelationResult:
        """Predict spatial relation for one pair from bbox geometry."""
        img_w, img_h = img.size
        mr_a = self._mc.compute(img_np, box_a, label_a)
        mr_b = self._mc.compute(img_np, box_b, label_b)

        spatial = self._geo.compute(
            box_a, mr_a.centroid, box_b, mr_b.centroid,
            img_w, img_h, label_a, label_b,
            count_a=count_a, count_b=count_b,
        )
        logger.info(f"Spatial [{label_a} | {label_b}] → '{spatial}'")

        return PairRelationResult(
            from_obj=label_a, to_obj=label_b,
            spatial=spatial, confidence=0.0,
        )

    def run_batch(self) -> list:
        """Run prediction on all images."""
        logger.info("=" * 60)
        logger.info("Starting Relation Prediction Pipeline")
        logger.info("=" * 60)
        
        
        # Load detection results
        with open(self.cfg.detection_json, encoding="utf-8") as f:
            reports = json.load(f)

        # Build prompt-order lookup: prompt_id → set of (subject_name, object_name)
        prompt_order: dict = {}
        if os.path.isfile(self.cfg.parsed_prompts_json):
            with open(self.cfg.parsed_prompts_json, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = rec.get("id", "")
                    id_to_name = {o["id"]: o["name"] for o in rec.get("objects", [])}
                    pairs: set = set()
                    for rel in rec.get("relations", []):
                        subj = id_to_name.get(rel.get("subject", ""), "")
                        obj  = id_to_name.get(rel.get("object", ""), "")
                        if subj and obj:
                            pairs.add((subj, obj))
                    prompt_order[pid] = pairs

        # Build attribute lookup
        attr_lookup = {}
        if os.path.isfile(self.cfg.attribute_json):
            with open(self.cfg.attribute_json, encoding="utf-8") as f:
                for entry in json.load(f):
                    pid = entry.get("prompt_id", "")
                    attr_lookup[pid] = {
                        o["label"]: {k: v for k, v in o.items() if k != "label"}
                        for o in entry.get("objects", [])
                    }

        results = []
        collision_filtered_count = 0

        for idx, report in enumerate(reports):
            pid = report["prompt_id"]
            image_path = report.get("image_path", "")
            correct = report.get("correct", [])

            logger.info(f"\nProcessing {idx+1}/{len(reports)}: {pid}")

            dropped_labels = {d["label"] for d in report.get("collision_dropped", [])}
            if dropped_labels:
                correct = [c for c in correct if c.get("label") not in dropped_labels]
                collision_filtered_count += 1
                logger.info(f"  Collision: dropped {dropped_labels}, winner kept")

            # Single detected object — write SG entry with 1 node, no edges
            if len(correct) == 1:
                logger.info(f"  Single object detected — writing node-only SG entry")
                pid_attrs = attr_lookup.get(pid, {})
                lbl = correct[0].get("label", "")
                node = {"label": lbl}
                obj_attr = pid_attrs.get(lbl, {})
                if "count" in obj_attr:
                    node["count"] = obj_attr["count"]
                attrs = {k: v for k, v in obj_attr.items() if k != "count"}
                if attrs:
                    node["attributes"] = attrs
                results.append({
                    "prompt_id":  pid,
                    "prompt":     report.get("prompt", ""),
                    "image_path": image_path,
                    "nodes":      [node],
                    "edges":      [],
                })
                continue

            if len(correct) == 0:
                logger.info(f"  No objects detected — writing empty SG entry")
                results.append({
                    "prompt_id":  pid,
                    "prompt":     report.get("prompt", ""),
                    "image_path": image_path,
                    "nodes":      [],
                    "edges":      [],
                })
                continue

            try:
                img = Image.open(image_path).convert("RGB")
                img_np = np.array(img)
            except Exception as exc:
                logger.warning(f"Cannot open {image_path}: {exc}")
                continue

            count_map = {
                tuple(o["bbox"]): o.get("count", 1)
                for o in report.get("objects_detected", [])
                if "bbox" in o
            }

            pid_prompt_pairs = prompt_order.get(pid, set())
            edges = []
            for i in range(len(correct)):
                for j in range(i + 1, len(correct)):
                    oa = correct[i]
                    ob = correct[j]
                    la = oa.get("label", "")
                    lb = ob.get("label", "")
                    ba = oa.get("bbox")
                    bb = ob.get("bbox")
                    if not (la and lb and ba and bb):
                        continue

                    ca = count_map.get(tuple(ba), 1)
                    cb = count_map.get(tuple(bb), 1)

                    # Preserve subject→object order from the original prompt.
                    if (lb, la) in pid_prompt_pairs and (la, lb) not in pid_prompt_pairs:
                        la, lb = lb, la
                        ba, bb = bb, ba
                        ca, cb = cb, ca

                    pred = self.predict_pair(
                        img, img_np, la, ba, lb, bb,
                        count_a=ca, count_b=cb,
                    )

                    spatial    = pred.spatial
                    confidence = pred.confidence
                    from_obj   = pred.from_obj
                    to_obj     = pred.to_obj

                    # Normalize "under"
                    if spatial == "under":
                        from_obj, to_obj = to_obj, from_obj
                        spatial = "on"

                    edge = {
                        "from":       from_obj,
                        "to":         to_obj,
                        "spatial":    spatial,
                        "confidence": round(confidence, 4),
                    }
                    edges.append(edge)

                    logger.info(
                        f"  Edge [{from_obj}→{to_obj}] spatial={spatial} conf={confidence:.4f}"
                    )

            pid_attrs = attr_lookup.get(pid, {})
            nodes = []
            for o in correct:
                lbl = o.get("label", "")
                node = {"label": lbl}
                obj_attr = pid_attrs.get(lbl, {})
                if "count" in obj_attr:
                    node["count"] = obj_attr["count"]
                attrs = {k: v for k, v in obj_attr.items() if k != "count"}
                if attrs:
                    node["attributes"] = attrs
                nodes.append(node)

            results.append({
                "prompt_id": pid,
                "prompt": report.get("prompt", ""),
                "image_path": image_path,
                "nodes": nodes,
                "edges": edges,
            })

        out_path = os.path.join(self.cfg.output_dir, "scenegraph.json")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*58}")
        print(f"  Relation Prediction Results")
        print(f"  Processed                : {len(results)}")
        print(f"  Collision (winner kept)  : {collision_filtered_count}")
        print(f"  Saved → {out_path}")
        print(f"{'='*58}\n")
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # Create vocab files if they don't exist
    os.makedirs("vocab", exist_ok=True)
    
    # Create spatial.txt if it doesn't exist
    spatial_path = "vocab/spatial.txt"
    if not os.path.exists(spatial_path):
        with open(spatial_path, 'w') as f:
            f.write("# Background surfaces for spatial relations\n")
            f.write("wall\nfloor\nground\nceiling\ngrass\nroad\n")
            f.write("pavement\nsidewalk\nsky\nlawn\nfield\ndirt\n")
            f.write("carpet\nmat\ntarmac\nasphalt\nconcrete\ntile\n")
            f.write("table\ndesk\nsurface\nplatform\nshelf\ncountertop\n")
            f.write("bed\nsofa\ncouch\ncharger\npad\ntray\nplate\ndock\nstand\n")
        print(f"Created {spatial_path}")
    
    
    # Run the predictor
    predictor = RelationPredictor()
    predictor.run_batch()