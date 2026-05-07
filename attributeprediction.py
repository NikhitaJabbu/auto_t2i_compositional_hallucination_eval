import os
import json
import logging
import glob as _glob
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ==========================================================
# CONFIG
# ==========================================================
@dataclass
class AttributeConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Background-scale detection
    bg_area_ratio: float         = 0.70
    bg_span_ratio: float         = 0.85
    bg_patch_frac: float         = 0.25
    bg_patch_min_px: int         = 64
    # Skip shape when another object covers this fraction of bbox
    overlap_bleed_threshold: float = 0.35
    # Semantic normalisation — similarity threshold for replacing predicted with prompt word
    normalise_threshold: float   = 0.82


# ==========================================================
# HELPERS
# ==========================================================
_MODIFIER_WORDS = {
    "the", "a", "an", "big", "small", "large", "tiny", "old", "new",
    "dark", "light", "bright", "pale", "soft", "hard", "thick", "thin",
    "tall", "short", "long", "wide", "narrow", "dining", "coffee",
    "grey", "curved", "striped", "polished", "painted", "matte", "glossy",
}
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
}


def _extract_base_noun(label: str) -> str:
    words = label.lower().split()
    for w in reversed(words):
        if w not in _MODIFIER_WORDS and not w.isdigit() and w not in _NUMBER_WORDS:
            return w
    return words[-1] if words else label.lower()


def _safe_name(text: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip().lower())
    out = "_".join(filter(None, out.split("_")))
    return out[:80] if out else "obj"


# Labels that cannot meaningfully have a "shape" attribute
_NO_SHAPE_LABELS = {
    "hand", "hands", "person", "people", "man", "woman", "girl", "boy",
    "surface", "floor", "wall", "ground", "ceiling", "sky", "background",
    "counter", "platform", "road", "street",
}

# Labels that cannot meaningfully have a fabric/synthetic "material" attribute.
# Living creatures and natural surfaces get RAM++ tags from surroundings (velvet
# cushion behind cat → cat=velvet) so we block material detection for them.
_NO_MATERIAL_LABELS = {
    # Animals / living
    "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "rabbit",
    "animal", "pet", "person", "people", "man", "woman", "child", "baby",
    # Natural outdoor surfaces
    "grass", "lawn", "tree", "bush", "plant", "flower", "leaf",
    "sky", "cloud", "water", "sand", "soil", "dirt", "mud",
    # Abstract / semantic
    "light", "shadow", "reflection", "background",
}


def _is_bg_scale(bbox: list, img_w: int, img_h: int,
                 area_ratio: float = 0.50,
                 span_ratio: float = 0.70) -> bool:
    """
    True when the bbox is background-scale (wall, floor, ceiling).
    Two criteria — either alone is sufficient:
      1. Area covers >area_ratio of the image (default 0.70)
      2. Spans >span_ratio in BOTH width AND height simultaneously
         (a car filling 95% of width but only 50% of height is NOT bg-scale)
    """
    w    = bbox[2] - bbox[0]
    h    = bbox[3] - bbox[1]
    area = w * h
    return (area / max(img_w * img_h, 1) > area_ratio or
            (w / img_w > span_ratio and h / img_h > span_ratio))


def _center_patch(img: Image.Image, bbox: list,
                  other_bboxes: list = None,
                  patch_frac: float = 0.25,
                  min_px: int = 64) -> Image.Image:
    """
    For background-scale objects (surface, floor, wall) sample a small patch
    from the corner that has the least overlap with other detected objects.

    Tries bottom-left, bottom-right, top-left, top-right, centre — picks
    whichever avoids sphere/robot/object shadows and colours bleeding in.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pw = max(int((x2 - x1) * patch_frac), min_px)
    ph = max(int((y2 - y1) * patch_frac), min_px)

    candidates = [
        (x1,               y2 - ph,           x1 + pw, y2),
        (x2 - pw,          y2 - ph,           x2,      y2),
        (x1,               y1,                x1 + pw, y1 + ph),
        (x2 - pw,          y1,                x2,      y1 + ph),
        ((x1+x2)//2-pw//2, (y1+y2)//2-ph//2,
         (x1+x2)//2+pw//2, (y1+y2)//2+ph//2),
    ]

    def _overlap(p):
        if not other_bboxes:
            return 0.0
        px1c, py1c, px2c, py2c = p
        total = 0.0
        for ob in other_bboxes:
            if ob is None or len(ob) < 4:
                continue
            ix1 = max(px1c, ob[0]); ix2 = min(px2c, ob[2])
            iy1 = max(py1c, ob[1]); iy2 = min(py2c, ob[3])
            total += max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return total

    best = min(candidates, key=_overlap)
    bx1  = max(0, best[0]); by1 = max(0, best[1])
    bx2  = min(img.width, best[2]); by2 = min(img.height, best[3])
    if bx2 <= bx1 or by2 <= by1:
        return img.crop((x1, y1, x2, y2))
    return img.crop((bx1, by1, bx2, by2))


def _tile_to_square(img: Image.Image, max_aspect: float = 5.0) -> Image.Image:
    """Tile extreme-aspect-ratio crops along the short axis until roughly square.
    Preserves surface texture (marble veins, fabric weave) without adding unrelated context."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if max(w, h) / max(min(w, h), 1) <= max_aspect:
        return img

    if w >= h:
        # wide strip — tile vertically until height ≥ width
        n_tiles = int(w / max(h, 1)) + 1
        canvas  = Image.new("RGB", (w, h * n_tiles))
        for i in range(n_tiles):
            canvas.paste(img, (0, i * h))
        side = min(w, h * n_tiles)
        return canvas.crop((0, 0, side, side))
    else:
        # tall strip — tile horizontally until width ≥ height
        n_tiles = int(h / max(w, 1)) + 1
        canvas  = Image.new("RGB", (w * n_tiles, h))
        for i in range(n_tiles):
            canvas.paste(img, (i * w, 0))
        side = min(h, w * n_tiles)
        return canvas.crop((0, 0, side, side))


def _overlap_fraction(bbox: list, other_bboxes: list) -> float:
    """
    Return the fraction of bbox that is covered by any of the other_bboxes.
    Used to detect attribute bleeding (e.g. hand crop contains 62% sphere).
    """
    bx1, by1, bx2, by2 = bbox
    area = max((bx2 - bx1) * (by2 - by1), 1)
    covered = 0.0
    for ob in other_bboxes:
        if ob is None or len(ob) < 4:
            continue
        ix1 = max(bx1, ob[0]);  ix2 = min(bx2, ob[2])
        iy1 = max(by1, ob[1]);  iy2 = min(by2, ob[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        covered += inter
    return min(covered / area, 1.0)


def _max_other_contained(bbox: list, other_bboxes: list) -> float:
    """
    Return the maximum fraction of any single OTHER bbox that falls inside
    this bbox.  High value means a neighbouring object is largely enclosed
    within the current crop and its colour will bleed into the crop.

    Example: towel (blue) is 69 % inside bathtub bbox → returns 0.69.
    This is different from _overlap_fraction which measures how much of the
    CURRENT object is covered, not how much of the OTHER object is inside it.
    """
    bx1, by1, bx2, by2 = bbox
    max_frac = 0.0
    for ob in other_bboxes:
        if ob is None or len(ob) < 4:
            continue
        ix1 = max(bx1, ob[0]);  ix2 = min(bx2, ob[2])
        iy1 = max(by1, ob[1]);  iy2 = min(by2, ob[3])
        inter    = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_ob  = max((ob[2] - ob[0]) * (ob[3] - ob[1]), 1)
        max_frac = max(max_frac, inter / area_ob)
    return max_frac


def _find_crop(crop_dir: str, label: str) -> Optional[str]:
    """Find the saved crop. Crops are named simply as 'chair.png'."""
    base_noun  = _extract_base_noun(label)
    candidates = [_safe_name(label), _safe_name(base_noun)]

    for c in list(candidates):
        if c.endswith("s") and len(c) > 3:
            candidates.append(c[:-1])
        else:
            candidates.append(c + "s")

    for name in candidates:
        path = os.path.join(crop_dir, f"{name}.png")
        if os.path.isfile(path):
            return path

    # fallback: any png whose stem overlaps with a candidate
    for path in _glob.glob(os.path.join(crop_dir, "*.png")):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if any(c in stem or stem in c for c in candidates if c):
            return path

    return None


# ==========================================================
# ATTRIBUTE PREDICTOR — Qwen2-VL 7B
# ==========================================================
class _Qwen2VLPredictor:
    """Qwen2-VL 7B for direct attribute prediction from crop images."""

    def __init__(self, cfg: AttributeConfig):
        self.cfg                    = cfg
        self._model                 = None
        self._processor             = None
        self._process_vision_info   = None
        self._ready: Optional[bool] = None

    def _load(self) -> bool:
        """Load Qwen2-VL 7B for direct attribute prediction from crop images."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
            self._processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16
            ).to(self.cfg.device).eval()
            logger.info("Qwen2-VL 7B loaded for attribute prediction")
            return True
        except Exception as exc:
            logger.warning("Qwen2-VL load failed: %s", exc)
            return False

    @property
    def ready(self) -> bool:
        if self._ready is None:
            self._ready = self._load()
        return self._ready

    def _parse(self, response: str) -> dict:
        """Parse 'color=X / material=Y / shape=Z' lines from model response."""
        attrs = {}
        for line in response.strip().splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip().lower()
                val = val.strip().lower().split()[0] if val.strip() else ""
                if key in ("color", "material", "shape") and val:
                    attrs[key] = val
        return attrs

    def predict(self, image: Image.Image, label: str = "") -> dict:
        """Predict color, material, shape directly from a crop image."""
        if not self.ready:
            return {}
        try:
            subject = f"the {label}" if label else "the main object"
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Look at {subject} in this image.\n"
                    "Output ONLY in this exact format:\n"
                    "color=X\n"
                    "material=Y\n"
                    "shape=Z\n"
                    "One word per attribute. Describe what you actually see."
                )},
            ]}]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = self._process_vision_info(messages)
            inputs = self._processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(self.cfg.device)
            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=30)
            response = self._processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return self._parse(response)
        except Exception as exc:
            logger.warning("Qwen2-VL prediction failed: %s", exc)
            return {}


# ==========================================================
# SEMANTIC NORMALISER
# ==========================================================
class _SemanticNormaliser:
    """Replaces a predicted attribute with the prompt's expected word when they are
    semantically equivalent (cosine similarity ≥ threshold) but textually different.
    Only operates on categories that the prompt explicitly annotates."""

    def __init__(self, threshold: float, device: str):
        self.threshold          = threshold
        self.device             = device
        self._model             = None
        self._tokenizer         = None
        self._cache: dict       = {}

    def _load(self) -> bool:
        try:
            import open_clip
            self._model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self._model   = self._model.to(self.device).eval()
            self._tokenizer = open_clip.get_tokenizer("ViT-L-14")
            return True
        except Exception as exc:
            logger.warning("Semantic normaliser load failed: %s", exc)
            return False

    def _embed(self, word: str) -> Optional[torch.Tensor]:
        if word in self._cache:
            return self._cache[word]
        if self._model is None and not self._load():
            return None
        tokens = self._tokenizer([word]).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        self._cache[word] = feat
        return feat

    def normalise(self, predicted: str, expected: str, attr_type: str) -> str:
        """Return expected if predicted is semantically close enough, else keep predicted."""
        if predicted == expected:
            return predicted
        pred_emb = self._embed(predicted)
        exp_emb  = self._embed(expected)
        if pred_emb is None or exp_emb is None:
            return predicted
        sim = (pred_emb @ exp_emb.T).item()
        logger.debug("Normalise [%s] '%s' vs '%s' sim=%.3f", attr_type, predicted, expected, sim)
        return expected if sim >= self.threshold else predicted


# ==========================================================
# ATTRIBUTE STAGE
# ==========================================================
class AttributeStage:
    def __init__(self, cfg: AttributeConfig = None):
        self.cfg         = cfg or AttributeConfig()
        self._predictor  = _Qwen2VLPredictor(self.cfg)
        self._normaliser = _SemanticNormaliser(self.cfg.normalise_threshold, self.cfg.device)

    def _detect(self, crop: Image.Image, label: str = "", skip_categories: set = None) -> dict:
        """Predict attributes from crop, dropping skipped categories."""
        skip  = skip_categories or set()
        attrs = self._predictor.predict(crop, label)
        for cat in skip:
            attrs.pop(cat, None)
        return attrs

    def run_prompt(self, prompt_id: str, correct_objects: list,
                   crops_dir: str, image_path: str = "",
                   expected_attrs: dict = None) -> dict:
        # expected_attrs: {object_name: {attr_type: value}}  — from parsed_prompts.jsonl
        # e.g. {"bathtub": {"color": "blue"}, "towel": {"color": "white"}}
        objects_out = []

        # Load full image once — needed for bbox crops and dimension checks
        full_img: Optional[Image.Image] = None
        img_w, img_h = 1024, 1024
        if image_path and os.path.isfile(image_path):
            try:
                full_img = Image.open(image_path).convert("RGB")
                img_w, img_h = full_img.size
            except Exception:
                full_img = None

        # Collect all bboxes to detect attribute bleeding later
        all_bboxes = [
            (item.get("bbox") if isinstance(item, dict) else None)
            for item in correct_objects
        ]

        for idx, item in enumerate(correct_objects):
            label = item["label"] if isinstance(item, dict) else item
            bbox  = item.get("bbox") if isinstance(item, dict) else None

            entry        = {"label": _extract_base_noun(label)}
            crop         = None
            skip_cats: set = set()
            other_inside   = 0.0   # fraction of any neighbour bbox inside ours

            base_noun = _extract_base_noun(label)

            if full_img is not None and bbox and len(bbox) >= 4:
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(img_w, int(bbox[2])), min(img_h, int(bbox[3]))

                if x2 > x1 and y2 > y1:
                    # Build other_bboxes first — needed for both overlap
                    # detection (Fix 1) and corner-patch selection (Fix 2)
                    other_bboxes = [b for k, b in enumerate(all_bboxes) if k != idx]

                    bg = _is_bg_scale(bbox, img_w, img_h,
                                      area_ratio=self.cfg.bg_area_ratio,
                                      span_ratio=self.cfg.bg_span_ratio)

                    overlap       = _overlap_fraction(bbox, other_bboxes)
                    other_inside  = _max_other_contained(bbox, other_bboxes)

                    if bg:
                        crop = _center_patch(full_img, bbox,
                                            other_bboxes=other_bboxes,
                                            patch_frac=self.cfg.bg_patch_frac,
                                            min_px=self.cfg.bg_patch_min_px)
                        skip_cats.add("shape")
                    else:
                        # SAM crop is already a tight object mask — neighbour bleeding
                        # inside the bbox does not affect it. Always prefer SAM.
                        # Only fall back to _center_patch when no SAM crop exists.
                        sam_path = _find_crop(os.path.join(crops_dir, prompt_id), label)
                        if sam_path and os.path.isfile(sam_path):
                            crop = _tile_to_square(
                                Image.open(sam_path).convert("RGB")
                            )
                        elif other_inside >= 0.50:
                            crop = _center_patch(full_img, bbox,
                                                other_bboxes=other_bboxes,
                                                patch_frac=self.cfg.bg_patch_frac,
                                                min_px=self.cfg.bg_patch_min_px)
                        else:
                            crop = _tile_to_square(
                                full_img.crop((x1, y1, x2, y2))
                            )

                    if overlap > self.cfg.overlap_bleed_threshold:
                        skip_cats.add("shape")
                    # High overlap also bleeds material (fabric from neighbor
                    # confuses polyester/resin/velvet detection on hard objects)
                    if overlap > 0.50:
                        skip_cats.add("material")

                    # Semantic objects that have no meaningful shape attribute
                    if base_noun in _NO_SHAPE_LABELS:
                        skip_cats.add("shape")
            if base_noun in _NO_MATERIAL_LABELS:
                skip_cats.add("material")

            if crop is None:
                crop_path = _find_crop(os.path.join(crops_dir, prompt_id), label)
                if crop_path and os.path.isfile(crop_path):
                    crop = _tile_to_square(
                        Image.open(crop_path).convert("RGB")
                    )

            if crop is not None:
                attrs = self._detect(crop, label=base_noun, skip_categories=skip_cats)

                # Normalise only categories present in the prompt for this object
                obj_expected = (expected_attrs or {}).get(base_noun, {})
                for attr_type, predicted_val in list(attrs.items()):
                    if attr_type in obj_expected:
                        attrs[attr_type] = self._normaliser.normalise(
                            predicted_val, obj_expected[attr_type], attr_type
                        )

                entry.update(attrs)

            objects_out.append(entry)

        return {"prompt_id": prompt_id, "objects": objects_out}


# ==========================================================
# BATCH RUNNER
# ==========================================================
def _load_expected_attrs(parsed_prompts_path: str) -> dict:
    """Build lookup: {prompt_id → {object_name → {attr_type → value}}} from parsed_prompts.jsonl."""
    lookup = {}
    if not os.path.isfile(parsed_prompts_path):
        return lookup
    with open(parsed_prompts_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pid      = entry.get("id") or entry.get("prompt_id", "")
            objects  = {o["id"]: o["name"] for o in entry.get("objects", [])}
            obj_attrs: dict = {}
            for attr in entry.get("attributes", []):
                obj_name = objects.get(attr["obj"], "")
                if obj_name:
                    obj_attrs.setdefault(obj_name, {})[attr["type"]] = attr["value"]
            lookup[pid] = obj_attrs
    return lookup


def run_attribute_detection(
    detection_json: str = "Outputs/detection_results/detection_results.json",
    crops_dir: str = "Outputs/detection_results/crops",
    parsed_prompts: str = "Outputs/parsed_prompts.jsonl",
    collision_exclusions: dict = None,
    cfg: AttributeConfig = None,
) -> list:
    """Run attribute detection on all valid prompts using Qwen2-VL 7B.

    Skips collision prompts. Processes single-detected prompts (1 object found).
    Applies semantic normalisation where the prompt explicitly states an attribute.
    Saves attribute_results.json next to detection_results.json.
    """
    cfg = cfg or AttributeConfig()

    if not os.path.isfile(detection_json):
        print(f"ERROR: detection results not found at '{detection_json}'")
        return []

    with open(detection_json, encoding="utf-8") as f:
        reports = json.load(f)

    expected_lookup = _load_expected_attrs(parsed_prompts)

    # Build dropped-label lookup: pid -> set of labels that lost their collision pair
    if collision_exclusions is not None:
        collision_dropped_lookup = {
            pid: set(dropped_labels)
            for pid, dropped_labels in collision_exclusions.items()
        }
    else:
        collision_dropped_lookup = {
            r["prompt_id"]: {d["label"] for d in r.get("collision_dropped", [])}
            for r in reports
            if r.get("collision_dropped")
        }

    stage   = AttributeStage(cfg)
    results = []
    n_collision_filtered = 0

    for report in reports:
        pid     = report["prompt_id"]
        correct = report.get("correct", [])

        dropped_labels = collision_dropped_lookup.get(pid, set())
        if dropped_labels:
            correct = [c for c in correct if c.get("label") not in dropped_labels]
            n_collision_filtered += 1
        if not correct:
            results.append({
                "prompt_id": pid,
                "prompt":    report.get("prompt", ""),
                "objects":   [],
            })
            continue

        result = stage.run_prompt(
            pid, correct, crops_dir,
            image_path=report.get("image_path", ""),
            expected_attrs=expected_lookup.get(pid, {}),
        )

        # Add prompt text to output
        result["prompt"] = report.get("prompt", "")

        # Merge count from counting_results into each object entry
        count_lookup = {}
        for cr in report.get("counting_results", []):
            count_lookup[cr["label"].lower()] = cr["count"]

        for obj_entry in result["objects"]:
            obj_label = obj_entry["label"].lower()
            matched_count = count_lookup.get(obj_label)
            if matched_count is None:
                for clabel, ccount in count_lookup.items():
                    if obj_label in clabel or clabel in obj_label:
                        matched_count = ccount
                        break
            if matched_count is not None and matched_count > 0:
                obj_entry["count"] = matched_count

        results.append(result)

    # --- save one combined file next to detection_results.json ---
    out_dir  = os.path.dirname(os.path.abspath(detection_json))
    out_path = os.path.join(out_dir, "attribute_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Attribute detection complete  (Qwen2-VL 7B)")
    print(f"  Prompts processed         : {len(results)}")
    print(f"  Collision (winner kept)   : {n_collision_filtered}")
    print(f"  Saved → {out_path}")
    print(f"{'='*50}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attribute Detection — Qwen2-VL 7B")
    parser.add_argument("--detection-json",
                        default="Outputs/detection_results/detection_results.json")
    parser.add_argument("--crops-dir",      default="Outputs/detection_results/crops")
    parser.add_argument("--parsed-prompts", default="Outputs/parsed_prompts.jsonl")
    parser.add_argument("--device",         default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AttributeConfig(device=device)

    run_attribute_detection(
        detection_json=args.detection_json,
        crops_dir=args.crops_dir,
        parsed_prompts=args.parsed_prompts,
        cfg=cfg,
    )
