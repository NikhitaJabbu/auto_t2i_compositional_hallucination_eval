"""
experiment_runner.py — Latent size experiment orchestrator.

Runs the full hallucination evaluation pipeline for 5 different latent sizes.
For each size:
  1. Generate images via ComfyUI  (ImgGeneration logic)
  2. Detect objects               (ObjectDetection)
  3. Predict attributes           (attributeprediction)
  4. Predict spatial relations    (relations)
  5. Evaluate hallucinations      (hallucination_eval)

QA pairs (Outputs/prompts_qapairs.jsonl) and parsed prompts
(Outputs/parsed_prompts.jsonl) are shared across all sizes — generated once.

Results per size → Outputs/experiments/{W}x{H}/
Final comparison → Outputs/experiments/comparison_summary.json
"""

import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

COMFY_HOST       = "127.0.0.1"
COMFY_PORT       = 8188
COMFY            = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_OUTPUT_DIR = Path("ComfyUI/output")

WORKFLOW_PATH    = Path("Models/Sdxl_experiment.json")
PROMPTS_JSONL    = Path("Outputs/parsed_prompts.jsonl")
QA_PAIRS_JSONL   = Path("Outputs/prompts_qapairs.jsonl")
EXPERIMENTS_ROOT = Path("Outputs/experiments")

POSITIVE_NODE_ID = "6"   # shared CLIPTextEncode (positive)

# One entry per size: tag, SaveImage node ID, EmptyLatentImage node ID
SIZE_CONFIGS = [
    {"tag": "1024x1024", "save_node": "13", "latent_node": "10"},
    {"tag": "2048x2048", "save_node": "23", "latent_node": "20"},
    {"tag": "1048x256",  "save_node": "33", "latent_node": "30"},
    {"tag": "256x2048",  "save_node": "43", "latent_node": "40"},
    {"tag": "256x256",   "save_node": "53", "latent_node": "50"},
]

# ---------------------------------------------------------------------------
# Helpers — ComfyUI HTTP
# ---------------------------------------------------------------------------

def _http_json(method, url, payload=None, timeout=60):
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body.decode("utf-8")) if body else None


def _http_bytes(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _comfy_running():
    try:
        urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def _queue_prompt(workflow):
    client_id = str(uuid.uuid4())
    r = _http_json("POST", f"{COMFY}/prompt",
                   {"prompt": workflow, "client_id": client_id})
    if not r or "prompt_id" not in r:
        raise RuntimeError(f"Unexpected /prompt response: {r}")
    return r["prompt_id"]


def _wait_done(prompt_id, poll=1.0, timeout=900):
    t0 = time.time()
    while True:
        try:
            hist = _http_json("GET", f"{COMFY}/history/{prompt_id}",
                              timeout=30)
        except Exception:
            hist = None
        if isinstance(hist, dict):
            info = hist.get(prompt_id)
            if info:
                status = info.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(
                        f"ComfyUI error for {prompt_id}: "
                        f"{status.get('messages', [])}"
                    )
                if info.get("outputs"):
                    return info
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Timed out waiting for {prompt_id}")
        time.sleep(poll)


def _download_outputs(info, pid, size_to_imgdir):
    """
    Download all generated images and route each to the correct size directory.
    size_to_imgdir: {tag: Path} e.g. {"1024x1024": Path(...), ...}
    Matches by subfolder name (ComfyUI stores size tag as subfolder).
    """
    saved = []
    counters = {}
    for node_out in info.get("outputs", {}).values():
        if not isinstance(node_out, dict):
            continue
        for im in node_out.get("images", []):
            fname     = im.get("filename", "")
            subfolder = im.get("subfolder", "")
            ftype     = im.get("type", "output")
            if not fname:
                continue
            qs  = {"filename": fname, "subfolder": subfolder, "type": ftype}
            url = f"{COMFY}/view?{urllib.parse.urlencode(qs)}"
            data = _http_bytes(url)
            ext  = os.path.splitext(fname)[1] or ".png"
            # Route to the matching size dir via subfolder name
            out_dir = size_to_imgdir.get(subfolder)
            if out_dir is None:
                # Fallback: try matching by filename prefix fragment
                for tag, d in size_to_imgdir.items():
                    if tag.replace("x", "x") in fname or tag in subfolder:
                        out_dir = d
                        break
            if out_dir is None:
                out_dir = list(size_to_imgdir.values())[0]
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = counters.get(str(out_dir), 1)
            path = out_dir / f"{pid}_{idx:04d}{ext}"
            path.write_bytes(data)
            counters[str(out_dir)] = idx + 1
            saved.append(path)
            src = COMFY_OUTPUT_DIR / subfolder / fname
            try:
                src.unlink()
            except OSError:
                pass
    return saved


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------------------------
# Step 1 — Image generation (all 5 sizes per prompt in one ComfyUI call)
# ---------------------------------------------------------------------------

def generate_images(size_img_dirs: dict):
    """
    size_img_dirs: {tag: Path} — where to save images for each size.
    One ComfyUI queue submission per prompt generates all 5 sizes at once.
    """
    if not _comfy_running():
        print("  ERROR: ComfyUI is not running. Start it first:")
        print(f"         cd {Path('ComfyUI').resolve()} && python main.py --port {COMFY_PORT}")
        sys.exit(1)

    for d in size_img_dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    workflow_template = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    total, errors = 0, 0
    for rec in _load_jsonl(PROMPTS_JSONL):
        pid         = str(rec.get("id", "")).strip()
        prompt_text = str(rec.get("prompt", "")).strip()
        if not pid or not prompt_text:
            continue

        g = json.loads(json.dumps(workflow_template))  # deep copy

        # Set positive prompt (shared node)
        g[POSITIVE_NODE_ID]["inputs"]["text"] = prompt_text

        # Set each SaveImage prefix to  "<size_tag>/<pid>"
        for cfg in SIZE_CONFIGS:
            g[cfg["save_node"]]["inputs"]["filename_prefix"] = f"{cfg['tag']}/{pid}"

        try:
            print(f"  [{pid}] queuing all 5 sizes — {prompt_text[:60]}...")
            server_pid = _queue_prompt(g)
            info       = _wait_done(server_pid)
            saved      = _download_outputs(info, pid, size_img_dirs)
            print(f"  [{pid}] saved {len(saved)} images across all sizes")
            total += 1
        except Exception as e:
            print(f"  [{pid}] ERROR: {e}")
            errors += 1

    print(f"  Generation done — {total} ok, {errors} errors")

# ---------------------------------------------------------------------------
# Step 2 — Object detection
# ---------------------------------------------------------------------------

def run_detection(images_dir, detection_dir):
    from ObjectDetection import DetectionConfig, ObjectDetectionPipeline
    cfg = DetectionConfig()
    cfg.output_dir = str(detection_dir)
    pipeline = ObjectDetectionPipeline(cfg)
    pipeline.run_batch(
        str(PROMPTS_JSONL),
        str(images_dir),
        str(detection_dir),
    )

# ---------------------------------------------------------------------------
# Step 3 — Attribute prediction
# ---------------------------------------------------------------------------

def run_attributes(detection_dir):
    from attributeprediction import run_attribute_detection, AttributeConfig
    detection_json = str(detection_dir / "detection_results.json")
    crops_dir      = str(detection_dir / "crops")
    cfg = AttributeConfig()
    run_attribute_detection(
        detection_json=detection_json,
        crops_dir=crops_dir,
        cfg=cfg,
    )

# ---------------------------------------------------------------------------
# Step 4 — Spatial relation prediction
# ---------------------------------------------------------------------------

def run_relations(detection_dir):
    from relations import RelationPredictor, Config
    cfg = Config()
    cfg.detection_json      = str(detection_dir / "detection_results.json")
    cfg.attribute_json      = str(detection_dir / "attribute_results.json")
    cfg.parsed_prompts_json = str(PROMPTS_JSONL)
    cfg.output_dir          = str(detection_dir)
    RelationPredictor(cfg).run_batch()

# ---------------------------------------------------------------------------
# Step 5 — Hallucination evaluation
# ---------------------------------------------------------------------------

def run_hallucination(detection_dir, halluc_dir):
    import hallucination_eval as heval
    from pathlib import Path as _Path

    halluc_dir  = _Path(halluc_dir)
    halluc_dir.mkdir(parents=True, exist_ok=True)

    qa_raw   = list(heval.load_jsonl(_Path(QA_PAIRS_JSONL)))
    sg_list  = heval.load_json(_Path(detection_dir / "scenegraph.json"))
    det_list = heval.load_json(_Path(detection_dir / "detection_results.json"))

    sg_by_pid  = {heval.safe_get_prompt_id(e): e for e in sg_list}
    det_by_pid = {heval.safe_get_prompt_id(e): e for e in det_list}

    all_scores    = []
    all_qa_output = []

    for record in qa_raw:
        pid      = heval.safe_get_prompt_id(record)
        qa_items = heval.extract_qa_items(record)
        if not qa_items:
            continue
        sg_entry  = sg_by_pid.get(pid)  or {}
        det_entry = det_by_pid.get(pid) or {}
        scores, qa_recs = heval.score_prompt(pid, qa_items, sg_entry, det_entry)
        all_scores.append(scores)
        all_qa_output.append({
            "prompt_id":      pid,
            "prompt":         record.get("prompt", ""),
            "qa_comparisons": qa_recs,
        })

    summary = heval.compute_dataset_summary(all_scores)

    _save_json(all_qa_output, halluc_dir / "scenegraph_qa_answers.json")
    _save_json(all_scores,    halluc_dir / "hallucination_per_prompt.json")
    _save_json(summary,       halluc_dir / "hallucination_summary.json")

    return summary

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Build per-size directory map
    exp_dirs = {
        cfg["tag"]: {
            "images":    EXPERIMENTS_ROOT / cfg["tag"] / "images",
            "detection": EXPERIMENTS_ROOT / cfg["tag"] / "detection",
            "halluc":    EXPERIMENTS_ROOT / cfg["tag"] / "hallucination",
        }
        for cfg in SIZE_CONFIGS
    }

    # --- Step 1: Generate all sizes in one pass (one ComfyUI call per prompt) ---
    print(f"\n{'='*62}")
    print(f"  STEP 1 — Generating images (all 5 sizes)")
    print(f"{'='*62}")
    size_img_dirs = {cfg["tag"]: exp_dirs[cfg["tag"]]["images"] for cfg in SIZE_CONFIGS}
    generate_images(size_img_dirs)

    # --- Steps 2-5: Run full eval pipeline per size ---
    all_summaries = []

    for cfg in SIZE_CONFIGS:
        tag           = cfg["tag"]
        images_dir    = exp_dirs[tag]["images"]
        detection_dir = exp_dirs[tag]["detection"]
        halluc_dir    = exp_dirs[tag]["halluc"]

        print(f"\n{'='*62}")
        print(f"  EVALUATING: {tag}")
        print(f"{'='*62}")

        print(f"\n[2/5] Object detection...")
        run_detection(images_dir, detection_dir)

        print(f"\n[3/5] Attribute prediction...")
        run_attributes(detection_dir)

        print(f"\n[4/5] Spatial relations...")
        run_relations(detection_dir)

        print(f"\n[5/5] Hallucination eval...")
        summary = run_hallucination(detection_dir, halluc_dir)
        summary["latent_size"] = tag
        all_summaries.append(summary)

        print(f"\n  {tag} results:")
        print(f"    H_obj       : {summary['avg_H_obj']:.4f}")
        print(f"    H_attr      : {summary['avg_H_attr']:.4f}")
        print(f"    H_rel       : {summary['avg_H_rel']:.4f}")
        print(f"    H_obj_extra : {summary['avg_H_obj_extra']:.4f}")

    # --- Final comparison ---
    comparison_path = EXPERIMENTS_ROOT / "comparison_summary.json"
    _save_json(all_summaries, comparison_path)

    print(f"\n{'='*62}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'─'*62}")
    print(f"  {'Size':<12}  {'H_obj':>8}  {'H_attr':>8}  {'H_rel':>8}  {'H_extra':>8}")
    print(f"{'─'*62}")
    for s in all_summaries:
        print(
            f"  {s['latent_size']:<12}  "
            f"{s['avg_H_obj']:>8.4f}  "
            f"{s['avg_H_attr']:>8.4f}  "
            f"{s['avg_H_rel']:>8.4f}  "
            f"{s['avg_H_obj_extra']:>8.4f}"
        )
    print(f"{'='*62}")
    print(f"  Comparison saved → {comparison_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
