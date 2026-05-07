"""
run.py — Unified pipeline runner.

Select a model then a mode to execute:
  0. Regular      — full pipeline (baseline)
  1. Experiment 1 — 5 latent sizes, per-size hallucination comparison
  2. Experiment 2 — steps & CFG sensitivity
  3. Experiment 3 — 20 seeds per prompt, per-prompt hallucination analysis

Output structure:
  Outputs/
    shared/                        ← parsed prompts & QA pairs (shared across models)
    models/{model}/
      baseline/
      experiment_1/
      experiment_2/
      experiment_3/
    comparison/                    ← cross-model summary
"""

import json
import logging
import os
import sys
from pathlib import Path

MODEL_WORKFLOWS = {
    "sdxl": {
        "baseline":   Path("Models/Sdxl_imageGeneration.json"),
        "experiment": Path("Models/Sdxl_experiment.json"),
    },
    "flux_schnell": {
        "baseline":   Path("Models/Flux_schnell_imageGeneration.json"),
        "experiment": Path("Models/Flux_schnell_experiment.json"),
    },
    "sd15": {
        "baseline":   Path("Models/Sdxl_imageGeneration.json"),
        "experiment": Path("Models/Sdxl_experiment.json"),
    },
    "sd35": {
        "baseline":   Path("Models/Sd35_imageGeneration.json"),
        "experiment": Path("Models/Sd35_experiment.json"),
    },
}

MODEL_NODE_CONFIGS = {
    "sdxl":         {"ksampler_baseline": "3", "saveimage_baseline": "9"},
    "flux_schnell": {"ksampler_baseline": "7", "saveimage_baseline": "9"},
    "sd15":         {"ksampler_baseline": "3", "saveimage_baseline": "9"},
    "sd35":         {"ksampler_baseline": "3", "saveimage_baseline": "9"},
}

# Node-level overrides applied at runtime — avoids duplicate workflow JSON files.
# "baseline" patches → Sdxl_imageGeneration.json (nodes 3=KSampler, 4=Checkpoint, 5=Latent)
# "experiment" patches → Sdxl_experiment.json   (node 4=Checkpoint, 11/21/31/41/51=KSamplers)
MODEL_PATCH_CONFIGS = {
    "sdxl":         {"baseline": {}, "experiment": {}},
    "flux_schnell": {"baseline": {}, "experiment": {}},
    "sd15": {
        "baseline": {
            "4": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
            "3": {"cfg": 7.0},
        },
        "experiment": {
            "4":  {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
            "11": {"cfg": 7.0}, "21": {"cfg": 7.0}, "31": {"cfg": 7.0},
            "41": {"cfg": 7.0}, "51": {"cfg": 7.0},
        },
    },
    "sd35": {"baseline": {}, "experiment": {}},
}

_shared_models = {}  # carries ModelLoader across steps within the same session

# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------

STEPS = {
    "parse":     "Parse prompts          (prompt_triples)   → parsed_prompts.jsonl",
    "qa":        "Generate QA pairs      (qa_generator)     → prompts_qapairs.jsonl",
    "images":    "Generate images        (ImgGeneration/ComfyUI)",
    "detect":    "Object detection       (ObjectDetection)",
    "attrs":     "Attribute prediction   (attributeprediction)",
    "relations": "Spatial relations      (relretry)",
    "halluc":    "Hallucination eval     (hallucination_eval)",
}

# ---------------------------------------------------------------------------
# Top-level output roots
# ---------------------------------------------------------------------------

MODELS_ROOT    = Path("Outputs/models")
SHARED_DIR     = Path("Outputs/shared")
COMPARISON_DIR = Path("Outputs/comparison")

# ---------------------------------------------------------------------------
# Available models and their ComfyUI workflows
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = list(MODEL_WORKFLOWS.keys())

# ---------------------------------------------------------------------------
# Mutable globals — populated by _set_model() before any mode runs
# ---------------------------------------------------------------------------

REGULAR             = {}
EXPERIMENT_ROOT     = None
EXPERIMENT_WORKFLOW = None
EXP2_ROOT           = None
EXP3_ROOT           = None

# ---------------------------------------------------------------------------
# Experiment configs — uniform node IDs across all models (workflow_builder)
# ---------------------------------------------------------------------------

SIZE_CONFIGS = [
    {"tag": "1024x1024", "save_node": "13", "latent_node": "10"},
    {"tag": "2048x2048", "save_node": "23", "latent_node": "20"},
    {"tag": "1048x256",  "save_node": "33", "latent_node": "30"},
    {"tag": "256x2048",  "save_node": "43", "latent_node": "40"},
    {"tag": "256x256",   "save_node": "53", "latent_node": "50"},
]

EXP2_KSAMPLER_NODE  = "7"
EXP2_SAVEIMAGE_NODE = "9"

EXP2_STEPS_CONFIGS = [
    {"tag": "steps_6",  "steps": 6,  "cfg": 8},
    {"tag": "steps_35", "steps": 35, "cfg": 8},
    {"tag": "steps_65", "steps": 65, "cfg": 8},
    {"tag": "steps_90", "steps": 90, "cfg": 8},
]

EXP2_CFG_CONFIGS = [
    {"tag": "cfg_4",  "steps": 20, "cfg": 4},
    {"tag": "cfg_17", "steps": 20, "cfg": 17},
    {"tag": "cfg_24", "steps": 20, "cfg": 24},
    {"tag": "cfg_35", "steps": 20, "cfg": 35},
]

EXP3_N_RUNS         = 20
EXP3_TOP_N          = 20
EXP3_BASE_SEED      = 1000
EXP3_KSAMPLER_NODE  = "7"
EXP3_SAVEIMAGE_NODE = "9"

# ---------------------------------------------------------------------------
# Model setup — builds all paths and creates directory structure
# ---------------------------------------------------------------------------

def _apply_patches(workflow_path: Path, patches: dict) -> dict:
    """Load a workflow JSON and apply model-specific node overrides, returning a dict."""
    import copy
    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    if "prompt" in wf and isinstance(wf["prompt"], dict):
        wf = wf["prompt"]
    if not patches:
        return wf
    wf = copy.deepcopy(wf)
    for node_id, fields in patches.items():
        if node_id in wf:
            wf[node_id]["inputs"].update(fields)
    return wf


def _set_model(model_name: str) -> None:
    """Populate all path globals and create the full folder structure for model_name."""
    global REGULAR, EXPERIMENT_ROOT, EXPERIMENT_WORKFLOW, EXP2_ROOT, EXP3_ROOT
    global EXP2_KSAMPLER_NODE, EXP2_SAVEIMAGE_NODE, EXP3_KSAMPLER_NODE, EXP3_SAVEIMAGE_NODE

    model_dir = MODELS_ROOT / model_name
    nd        = MODEL_NODE_CONFIGS[model_name]
    patches   = MODEL_PATCH_CONFIGS.get(model_name, {"baseline": {}, "experiment": {}})

    REGULAR = {
        "raw_prompts":   Path("data/promts_150.jsonl"),
        "parsed":        SHARED_DIR / "parsed_prompts.jsonl",
        "qa_pairs":      SHARED_DIR / "prompts_qapairs.jsonl",
        "images":        model_dir / "baseline" / "images",
        "detection":     model_dir / "baseline" / "detection",
        "hallucination": model_dir / "baseline" / "hallucination",
        "workflow":      _apply_patches(MODEL_WORKFLOWS[model_name]["baseline"],   patches["baseline"]),
    }

    EXPERIMENT_ROOT     = model_dir / "experiment_1"
    EXPERIMENT_WORKFLOW = _apply_patches(MODEL_WORKFLOWS[model_name]["experiment"], patches["experiment"])
    EXP2_ROOT           = model_dir / "experiment_2"
    EXP3_ROOT           = model_dir / "experiment_3"

    EXP2_KSAMPLER_NODE  = nd["ksampler_baseline"]
    EXP2_SAVEIMAGE_NODE = nd["saveimage_baseline"]
    EXP3_KSAMPLER_NODE  = nd["ksampler_baseline"]
    EXP3_SAVEIMAGE_NODE = nd["saveimage_baseline"]

    _create_model_structure(model_name)


def _create_model_structure(model_name: str) -> None:
    """Pre-create the full directory tree for the selected model."""
    model_dir = MODELS_ROOT / model_name

    # Shared inputs and cross-model comparison
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    for sub in ("images", "detection", "hallucination"):
        (model_dir / "baseline" / sub).mkdir(parents=True, exist_ok=True)

    # Experiment 1 — latent sizes
    for cfg in SIZE_CONFIGS:
        for sub in ("images", "detection", "hallucination"):
            (model_dir / "experiment_1" / cfg["tag"] / sub).mkdir(parents=True, exist_ok=True)

    # Experiment 2 — steps & CFG
    for cfg in EXP2_STEPS_CONFIGS + EXP2_CFG_CONFIGS:
        for sub in ("images", "detection", "hallucination"):
            (model_dir / "experiment_2" / cfg["tag"] / sub).mkdir(parents=True, exist_ok=True)

    # Experiment 3 — seed variance runs + aggregated
    for i in range(EXP3_N_RUNS):
        for sub in ("images", "detection", "hallucination"):
            (model_dir / "experiment_3" / f"run_{i+1:03d}" / sub).mkdir(parents=True, exist_ok=True)
    (model_dir / "experiment_3" / "aggregated").mkdir(parents=True, exist_ok=True)

    print(f"  Directory structure ready → {model_dir}/")

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _header(title):
    w = 62
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")


def _section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def _ask_steps(all_steps: list) -> list:
    """Ask user which steps to run. Returns list of selected step keys."""
    print("\n  Steps to run (press Enter to run ALL, or enter numbers to skip):")
    for i, (key, desc) in enumerate(all_steps, 1):
        exists = _step_done(key)
        done_mark = " [done]" if exists else ""
        print(f"    {i}. {desc}{done_mark}")

    print("\n  Enter step numbers to SKIP (e.g. 1,2) or press Enter to run all: ", end="")
    raw = input().strip()

    if not raw:
        return [k for k, _ in all_steps]

    try:
        skip = {int(x.strip()) for x in raw.split(",") if x.strip()}
    except ValueError:
        print("  Invalid input — running all steps.")
        return [k for k, _ in all_steps]

    selected = [k for i, (k, _) in enumerate(all_steps, 1) if i not in skip]
    skipped  = [k for i, (k, _) in enumerate(all_steps, 1) if i in skip]
    if skipped:
        print(f"  Skipping: {', '.join(skipped)}")
    return selected


def _step_done(key: str) -> bool:
    """Quick heuristic to check if a step's output already exists."""
    checks = {
        "parse":     REGULAR.get("parsed"),
        "qa":        REGULAR.get("qa_pairs"),
        "images":    REGULAR.get("images"),
        "detect":    REGULAR.get("detection") and REGULAR["detection"] / "detection_results.json",
        "attrs":     REGULAR.get("detection") and REGULAR["detection"] / "attribute_results.json",
        "relations": REGULAR.get("detection") and REGULAR["detection"] / "scenegraph.json",
        "halluc":    REGULAR.get("hallucination") and REGULAR["hallucination"] / "hallucination_summary.json",
    }
    p = checks.get(key)
    return p is not None and Path(p).exists()


def _confirm(msg: str) -> bool:
    print(f"\n  {msg} [y/N]: ", end="")
    return input().strip().lower() == "y"

# ---------------------------------------------------------------------------
# Individual pipeline steps — regular mode
# ---------------------------------------------------------------------------

def step_parse():
    import spacy
    from prompt_triples import load_all_vocabs, load_jsonl, extract_all, save_jsonl
    _section("Parsing prompts")
    nlp = spacy.load("en_core_web_sm")
    attr_vocabs, spatial = load_all_vocabs(Path("vocab"))
    results = []
    for rec in load_jsonl(REGULAR["raw_prompts"]):
        doc = nlp(rec["prompt"])
        extracted = extract_all(doc, attr_vocabs, spatial)
        parsed = {"id": rec["id"], "prompt": rec["prompt"], **extracted}
        if not parsed.get("attributes"):
            parsed.pop("attributes", None)
        if not parsed.get("relations"):
            parsed.pop("relations", None)
        results.append(parsed)
    save_jsonl(results, REGULAR["parsed"])
    print(f"  Parsed {len(results)} prompts → {REGULAR['parsed']}")


def step_qa():
    from qa_generator import process_jsonl_file
    _section("Generating QA pairs")
    process_jsonl_file(str(REGULAR["parsed"]), str(REGULAR["qa_pairs"]))


def step_images(workflow_path=None, out_dir=None, extra_node_overrides=None):
    import uuid, time, urllib.parse, urllib.request
    from ImgGeneration import (
        http_json, http_bytes, find_positive_node,
        find_clip_text_nodes, find_saveimage_nodes,
        queue_prompt, wait_for_done, load_jsonl,
    )
    _section("Generating images")

    wf_input = workflow_path if workflow_path is not None else REGULAR["workflow"]
    img_dir  = Path(out_dir or REGULAR["images"])
    img_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(wf_input, dict):
        workflow = wf_input
    else:
        workflow = json.loads(Path(wf_input).read_text(encoding="utf-8"))
        if "prompt" in workflow and isinstance(workflow["prompt"], dict):
            workflow = workflow["prompt"]

    pos_node   = find_positive_node(workflow)
    save_nodes = find_saveimage_nodes(workflow)
    all_txt    = find_clip_text_nodes(workflow) if not pos_node else None

    total, errors = 0, 0
    for rec in load_jsonl(REGULAR["raw_prompts"]):
        pid  = str(rec.get("id", "")).strip()
        text = str(rec.get("prompt", "")).strip()
        if not pid or not text:
            continue

        g = json.loads(json.dumps(workflow))
        if pos_node:
            g[pos_node]["inputs"]["text"] = text
        else:
            for nid in (all_txt or []):
                g[nid]["inputs"]["text"] = text

        if extra_node_overrides:
            for nid, fields in extra_node_overrides(pid).items():
                for field, val in fields.items():
                    g[nid]["inputs"][field] = val
        else:
            for nid in save_nodes:
                g[nid]["inputs"]["filename_prefix"] = pid

        try:
            print(f"  [{pid}] {text[:70]}...")
            spid  = queue_prompt(g)
            info  = wait_for_done(spid, timeout_s=1200)
            saved = _download_all(info, pid, img_dir, extra_node_overrides is not None)
            print(f"  [{pid}] → {len(saved)} image(s)")
            total += 1
        except Exception as e:
            print(f"  [{pid}] ERROR: {e}")
            errors += 1

    print(f"\n  Done — {total} ok, {errors} errors")


def _download_all(info, pid, img_dir, multi_size=False):
    import urllib.parse
    from ImgGeneration import http_bytes
    saved, idx = [], 1
    for node_out in info.get("outputs", {}).values():
        if not isinstance(node_out, dict):
            continue
        for im in node_out.get("images", []):
            fname     = im.get("filename", "")
            subfolder = im.get("subfolder", "")
            ftype     = im.get("type", "output")
            if not fname:
                continue
            qs   = {"filename": fname, "subfolder": subfolder, "type": ftype}
            url  = f"http://127.0.0.1:8188/view?{urllib.parse.urlencode(qs)}"
            data = http_bytes(url)
            ext  = os.path.splitext(fname)[1] or ".png"
            if multi_size and subfolder:
                dest = img_dir / subfolder / "images"
            else:
                dest = img_dir
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"{pid}_{idx:04d}{ext}"
            path.write_bytes(data)
            saved.append(path)
            idx += 1
    return saved


def step_detect(images_dir=None, detection_dir=None):
    from ObjectDetection import DetectionConfig, ObjectDetectionPipeline
    _section("Object detection")
    cfg = DetectionConfig()
    cfg.output_dir = str(detection_dir or REGULAR["detection"])
    pipeline = ObjectDetectionPipeline(cfg)
    pipeline.run_batch(
        str(REGULAR["parsed"]),
        str(images_dir or REGULAR["images"]),
        str(detection_dir or REGULAR["detection"]),
    )
    _shared_models['model_loader'] = pipeline.models


def step_attrs(detection_dir=None):
    from attributeprediction import run_attribute_detection, AttributeConfig
    _section("Attribute prediction")
    det_dir = Path(detection_dir or REGULAR["detection"])
    run_attribute_detection(
        detection_json=str(det_dir / "detection_results.json"),
        crops_dir=str(det_dir / "crops"),
        cfg=AttributeConfig(),
    )


def step_relations(detection_dir=None):
    from relretry import RelationPredictor, Config
    _section("Spatial relations")
    det_dir = Path(detection_dir or REGULAR["detection"])
    cfg = Config()
    cfg.detection_json      = str(det_dir / "detection_results.json")
    cfg.attribute_json      = str(det_dir / "attribute_results.json")
    cfg.parsed_prompts_json = str(REGULAR["parsed"])
    cfg.output_dir          = str(det_dir)
    ml = _shared_models.get('model_loader')
    sam_predictor = ml._sam if ml and ml._sam is not None else None
    RelationPredictor(cfg, sam_predictor=sam_predictor).run_batch()


def step_halluc(detection_dir=None, halluc_dir=None):
    import hallucination_eval as heval
    _section("Hallucination evaluation")
    det_dir = Path(detection_dir or REGULAR["detection"])
    out_dir = Path(halluc_dir    or REGULAR["hallucination"])

    summary = heval.run_evaluation(
        qa_pairs_path       = str(REGULAR["qa_pairs"]),
        scene_graph_path    = str(det_dir / "scenegraph.json"),
        detection_json_path = str(det_dir / "detection_results.json"),
        out_dir             = str(out_dir),
    )
    return summary


def _save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------------------------------------------------------------------------
# Mode 0 — Regular pipeline (baseline)
# ---------------------------------------------------------------------------

REGULAR_STEPS = [
    ("parse",     STEPS["parse"]),
    ("qa",        STEPS["qa"]),
    ("images",    STEPS["images"]),
    ("detect",    STEPS["detect"]),
    ("attrs",     STEPS["attrs"]),
    ("relations", STEPS["relations"]),
    ("halluc",    STEPS["halluc"]),
]

STEP_FN = {
    "parse":     step_parse,
    "qa":        step_qa,
    "images":    step_images,
    "detect":    step_detect,
    "attrs":     step_attrs,
    "relations": step_relations,
    "halluc":    step_halluc,
}

def _save_results():
    import update_results
    update_results.main()


def run_regular():
    import time
    _header("Mode 0 — Regular Pipeline (baseline)")
    selected = _ask_steps(REGULAR_STEPS)
    _timings = {}
    for key in selected:
        t0 = time.time()
        STEP_FN[key]()
        _timings[key] = time.time() - t0

    halluc_path = REGULAR["hallucination"] / "hallucination_summary.json"
    det_path    = REGULAR["detection"]     / "detection_summary.json"
    if halluc_path.exists():
        with open(halluc_path, encoding="utf-8") as f:
            hs = json.load(f)
        ds = {}
        if det_path.exists():
            with open(det_path, encoding="utf-8") as f:
                ds = json.load(f)
        _header("Regular Pipeline Results")
        print(f"  {'Metric':<30}  {'Value':>8}")
        print(f"  {'─'*30}  {'─'*8}")
        print(f"  {'H_obj  (semantic)':<30}  {hs.get('avg_H_obj',  0):>8.4f}")
        print(f"  {'H_attr (semantic)':<30}  {hs.get('avg_H_attr', 0):>8.4f}")
        print(f"  {'H_rel  (semantic)':<30}  {hs.get('avg_H_rel',  0):>8.4f}")
        if ds:
            print(f"  {'H_missed (detection)':<30}  {ds.get('H_obj_missed', 0):>8.4f}")
            print(f"  {'H_extra  (detection)':<30}  {ds.get('H_obj_extra',  0):>8.4f}")
        print(f"  {'Prompts evaluated':<30}  {hs.get('num_prompts', 0):>8}")

    if _timings:
        total_s   = sum(_timings.values())
        img_gen_s = _timings.get("images", 0)
        n_prompts = 150
        print(f"\n  {'─'*30}  {'─'*8}")
        print(f"  {'Latency (wall-clock)':<30}  {'':>8}")
        step_labels = {
            "parse": "Parse prompts", "qa": "QA generation",
            "images": "Image generation", "detect": "Object detection",
            "attrs": "Attribute prediction", "relations": "Spatial relations",
            "halluc": "Hallucination eval",
        }
        for key, elapsed in _timings.items():
            label = step_labels.get(key, key)
            print(f"  {'  ' + label:<30}  {elapsed:>7.1f}s")
        print(f"  {'─'*30}  {'─'*8}")
        print(f"  {'Total pipeline':<30}  {total_s:>7.1f}s")
        if img_gen_s > 0:
            print(f"  {'Avg image gen / prompt':<30}  {img_gen_s / n_prompts:>7.2f}s")

        latency_summary = {
            "total_s":               round(total_s, 1),
            "img_gen_s":             round(img_gen_s, 1),
            "img_gen_per_prompt_s":  round(img_gen_s / n_prompts, 2) if img_gen_s > 0 else None,
            "steps":                 {k: round(v, 1) for k, v in _timings.items()},
        }
        _save_json(latency_summary, REGULAR["hallucination"].parent / "latency_summary.json")
    _save_results()

# ---------------------------------------------------------------------------
# Mode 1 — Experiment 1 (5 latent sizes)
# ---------------------------------------------------------------------------

EXP1_STEPS = [
    ("images",    STEPS["images"]    + "  [all 5 sizes]"),
    ("detect",    STEPS["detect"]    + "  [per size]"),
    ("attrs",     STEPS["attrs"]     + "  [per size]"),
    ("relations", STEPS["relations"] + "  [per size]"),
    ("halluc",    STEPS["halluc"]    + "  [per size]"),
]

def run_experiment1():
    _header("Mode 1 — Experiment 1 (5 latent sizes)")
    print("\n  Sizes:", ", ".join(c["tag"] for c in SIZE_CONFIGS))
    print("  QA pairs reused from:", REGULAR["qa_pairs"])

    if not REGULAR["qa_pairs"].exists():
        print(f"\n  WARNING: QA pairs not found at {REGULAR['qa_pairs']}")
        print("  Run Mode 0 steps 1-2 first (parse + QA generation).")
        if not _confirm("Continue anyway?"):
            return

    selected = _ask_steps(EXP1_STEPS)

    if "images" in selected:
        def _overrides(pid):
            return {cfg["save_node"]: {"filename_prefix": f"{cfg['tag']}/{pid}"}
                    for cfg in SIZE_CONFIGS}

        step_images(
            workflow_path=EXPERIMENT_WORKFLOW,
            out_dir=EXPERIMENT_ROOT,
            extra_node_overrides=_overrides,
        )

    all_summaries = []
    for cfg in SIZE_CONFIGS:
        tag           = cfg["tag"]
        images_dir    = EXPERIMENT_ROOT / tag / "images"
        detection_dir = EXPERIMENT_ROOT / tag / "detection"
        halluc_dir    = EXPERIMENT_ROOT / tag / "hallucination"

        print(f"\n  ── {tag} ──")

        if "detect"    in selected: step_detect(images_dir, detection_dir)
        if "attrs"     in selected: step_attrs(detection_dir)
        if "relations" in selected: step_relations(detection_dir)
        if "halluc"    in selected:
            summary = step_halluc(detection_dir, halluc_dir)
            summary["latent_size"] = tag
            det_summary_path = detection_dir / "detection_summary.json"
            if det_summary_path.exists():
                with open(det_summary_path, encoding="utf-8") as f:
                    summary.update(json.load(f))
            all_summaries.append(summary)

    if all_summaries:
        comparison_path = EXPERIMENT_ROOT / "experiment_1_comparison.json"
        _save_json(all_summaries, comparison_path)

        _header("Experiment 1 Results")
        print(f"  {'Size':<12}  {'H_obj':>8}  {'H_attr':>8}  {'H_rel':>8}  {'H_missed':>10}  {'H_extra':>8}  {'#prompts':>10}  {'#rel':>6}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*6}")
        for s in all_summaries:
            print(
                f"  {s['latent_size']:<12}  "
                f"{s['avg_H_obj']:>8.4f}  "
                f"{s['avg_H_attr']:>8.4f}  "
                f"{s['avg_H_rel']:>8.4f}  "
                f"{s.get('H_obj_missed', float('nan')):>10.4f}  "
                f"{s.get('H_obj_extra',  float('nan')):>8.4f}  "
                f"{s['num_prompts']:>10}  "
                f"{s['num_rel_prompts']:>6}"
            )
        print(f"\n  Saved → {comparison_path}")
    _save_results()

# ---------------------------------------------------------------------------
# Mode 2 — Experiment 2 (steps & CFG sensitivity)
# ---------------------------------------------------------------------------

EXP2_MENU_STEPS = [
    ("images",    STEPS["images"]    + "  [all 8 configs]"),
    ("detect",    STEPS["detect"]    + "  [per config]"),
    ("attrs",     STEPS["attrs"]     + "  [per config]"),
    ("relations", STEPS["relations"] + "  [per config]"),
    ("halluc",    STEPS["halluc"]    + "  [per config]"),
]


def run_experiment2():
    _header("Mode 2 — Experiment 2 (Steps & CFG sensitivity)")
    print("\n  Steps sweep  — steps: 6, 35, 65, 90  |  CFG fixed at 8")
    print("  CFG sweep    — CFG: 4, 17, 24, 35    |  steps fixed at 20")
    print("  Resolution   — 1024×1024 (fixed)")
    print("  Seed         — fixed (from workflow)")
    print("  QA pairs reused from:", REGULAR["qa_pairs"])

    if not REGULAR["qa_pairs"].exists():
        print(f"\n  WARNING: QA pairs not found at {REGULAR['qa_pairs']}")
        print("  Run Mode 0 steps 1-2 first (parse + QA generation).")
        if not _confirm("Continue anyway?"):
            return

    selected = _ask_steps(EXP2_MENU_STEPS)

    all_configs  = EXP2_STEPS_CONFIGS + EXP2_CFG_CONFIGS
    _img_gen_times: dict = {}   # tag → seconds
    _exp2_t0 = __import__("time").time()

    if "images" in selected:
        for config in all_configs:
            import time as _time
            tag     = config["tag"]
            img_dir = EXP2_ROOT / tag / "images"
            print(f"\n  Generating images: {tag}  (steps={config['steps']}, cfg={config['cfg']})")

            def _make_overrides(cfg_steps, cfg_val):
                def _overrides(pid):
                    return {
                        EXP2_KSAMPLER_NODE:  {"steps": cfg_steps, "cfg": cfg_val},
                        EXP2_SAVEIMAGE_NODE: {"filename_prefix": pid},
                    }
                return _overrides

            _t0 = _time.time()
            step_images(
                workflow_path=REGULAR["workflow"],
                out_dir=img_dir,
                extra_node_overrides=_make_overrides(config["steps"], config["cfg"]),
            )
            _img_gen_times[tag] = _time.time() - _t0

    steps_summaries = []
    cfg_summaries   = []

    for group_configs, summaries in [
        (EXP2_STEPS_CONFIGS, steps_summaries),
        (EXP2_CFG_CONFIGS,   cfg_summaries),
    ]:
        for config in group_configs:
            tag           = config["tag"]
            images_dir    = EXP2_ROOT / tag / "images"
            detection_dir = EXP2_ROOT / tag / "detection"
            halluc_dir    = EXP2_ROOT / tag / "hallucination"

            print(f"\n  ── {tag}  (steps={config['steps']}, cfg={config['cfg']}) ──")

            if "detect"    in selected: step_detect(images_dir, detection_dir)
            if "attrs"     in selected: step_attrs(detection_dir)
            if "relations" in selected: step_relations(detection_dir)
            if "halluc"    in selected:
                summary = step_halluc(detection_dir, halluc_dir)
                summary["tag"]   = tag
                summary["steps"] = config["steps"]
                summary["cfg"]   = config["cfg"]
                det_summary_path = detection_dir / "detection_summary.json"
                if det_summary_path.exists():
                    with open(det_summary_path, encoding="utf-8") as f:
                        summary.update(json.load(f))
                summaries.append(summary)

    flat = steps_summaries + cfg_summaries
    if flat:
        comparison_path = EXP2_ROOT / "experiment_2_comparison.json"
        _save_json({"steps_sweep": steps_summaries, "cfg_sweep": cfg_summaries}, comparison_path)

        _header("Experiment 2 Results")
        for group_label, summaries in [
            ("Steps Sweep  (CFG=8 fixed)", steps_summaries),
            ("CFG Sweep  (steps=20 fixed)", cfg_summaries),
        ]:
            if not summaries:
                continue
            print(f"\n  {group_label}")
            print(f"  {'Tag':<12}  {'Steps':>6}  {'CFG':>6}  {'H_obj':>8}  {'H_attr':>8}  {'H_rel':>8}  {'H_missed':>10}  {'H_extra':>8}  {'#prompts':>10}  {'ImgGen(s)':>10}")
            print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")
            for s in summaries:
                img_t = _img_gen_times.get(s["tag"], float("nan"))
                print(
                    f"  {s['tag']:<12}  "
                    f"{s['steps']:>6}  "
                    f"{s['cfg']:>6.1f}  "
                    f"{s['avg_H_obj']:>8.4f}  "
                    f"{s['avg_H_attr']:>8.4f}  "
                    f"{s['avg_H_rel']:>8.4f}  "
                    f"{s.get('H_obj_missed', float('nan')):>10.4f}  "
                    f"{s.get('H_obj_extra',  float('nan')):>8.4f}  "
                    f"{s['num_prompts']:>10}  "
                    f"{img_t:>9.1f}s"
                )
        total_pipeline_s = __import__("time").time() - _exp2_t0
        print(f"\n  Total pipeline time (all 8 configs): {total_pipeline_s:.1f}s")
        print(f"  Saved → {comparison_path}")
    _save_results()

# ---------------------------------------------------------------------------
# Mode 3 — Experiment 3 (20 seeds per prompt)
# ---------------------------------------------------------------------------

EXP3_STEPS = [
    ("images",    STEPS["images"]    + "  [20 seeds × 150 prompts]"),
    ("detect",    STEPS["detect"]    + "  [per run]"),
    ("attrs",     STEPS["attrs"]     + "  [per run]"),
    ("relations", STEPS["relations"] + "  [per run]"),
    ("halluc",    STEPS["halluc"]    + "  [per run]"),
    ("aggregate", "Aggregate results  → aggregated/"),
]


def _aggregate_exp3(run_dirs, parsed_prompts_path, out_dir, top_n=20):
    import statistics

    complexity = {}
    if parsed_prompts_path.exists():
        with open(parsed_prompts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pid = str(rec.get("id", ""))
                complexity[pid] = {
                    "prompt":       rec.get("prompt", ""),
                    "n_objects":    len(rec.get("objects",    [])),
                    "n_attributes": len(rec.get("attributes", [])),
                    "n_relations":  len(rec.get("relations",  [])),
                }

    pid_runs = {}
    det_missed_vals, det_extra_vals = [], []
    for run_dir in run_dirs:
        path = run_dir / "hallucination" / "hallucination_per_prompt.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for s in json.load(f):
                pid = str(s.get("prompt_id", ""))
                pid_runs.setdefault(pid, []).append(s)
        det_path = run_dir / "detection" / "detection_summary.json"
        if det_path.exists():
            with open(det_path, encoding="utf-8") as f:
                ds = json.load(f)
            det_missed_vals.append(ds.get("H_obj_missed", 0))
            det_extra_vals.append(ds.get("H_obj_extra",  0))

    per_prompt = []
    for pid, runs in pid_runs.items():
        h_obj  = [r["H_obj"]  for r in runs]
        h_attr = [r["H_attr"] for r in runs]
        h_rel  = [r["H_rel"]  for r in runs]

        halluc_rate_obj  = sum(1 for r in runs if r["H_obj"]  > 0) / len(runs)
        halluc_rate_attr = sum(1 for r in runs if r["H_attr"] > 0) / len(runs)
        halluc_rate_rel  = sum(1 for r in runs if r["H_rel"]  > 0) / len(runs)

        consistency = round(
            1.0 - (statistics.stdev(h_obj) if len(h_obj) > 1 else 0.0), 4
        )

        per_prompt.append({
            "prompt_id":        pid,
            "n_runs":           len(runs),
            "avg_H_obj":        round(sum(h_obj)  / len(h_obj),  4),
            "avg_H_attr":       round(sum(h_attr) / len(h_attr), 4),
            "avg_H_rel":        round(sum(h_rel)  / len(h_rel),  4),
            "halluc_rate_obj":  round(halluc_rate_obj,  4),
            "halluc_rate_attr": round(halluc_rate_attr, 4),
            "halluc_rate_rel":  round(halluc_rate_rel,  4),
            "consistency":      consistency,
            **complexity.get(pid, {}),
        })

    per_prompt.sort(key=lambda x: x["prompt_id"])

    sort_key      = lambda x: (x["halluc_rate_obj"] + x["halluc_rate_attr"] + x["halluc_rate_rel"]) / 3
    sorted_by_rate = sorted(per_prompt, key=sort_key)
    best_prompts   = sorted_by_rate[:top_n]
    worst_prompts  = sorted_by_rate[-top_n:][::-1]

    N = len(per_prompt)
    summary = {
        "n_prompts":              N,
        "n_runs":                 len(run_dirs),
        "avg_H_obj":              round(sum(p["avg_H_obj"]  for p in per_prompt) / N, 4) if N else 0,
        "avg_H_attr":             round(sum(p["avg_H_attr"] for p in per_prompt) / N, 4) if N else 0,
        "avg_H_rel":              round(sum(p["avg_H_rel"]  for p in per_prompt) / N, 4) if N else 0,
        "avg_halluc_rate_obj":    round(sum(p["halluc_rate_obj"]  for p in per_prompt) / N, 4) if N else 0,
        "avg_halluc_rate_attr":   round(sum(p["halluc_rate_attr"] for p in per_prompt) / N, 4) if N else 0,
        "avg_halluc_rate_rel":    round(sum(p["halluc_rate_rel"]  for p in per_prompt) / N, 4) if N else 0,
        "consistently_bad_obj":   sum(1 for p in per_prompt if p["halluc_rate_obj"]  >= 0.8),
        "consistently_bad_attr":  sum(1 for p in per_prompt if p["halluc_rate_attr"] >= 0.8),
        "consistently_bad_rel":   sum(1 for p in per_prompt if p["halluc_rate_rel"]  >= 0.8),
        "consistently_good_obj":  sum(1 for p in per_prompt if p["halluc_rate_obj"]  <= 0.2),
        "consistently_good_attr": sum(1 for p in per_prompt if p["halluc_rate_attr"] <= 0.2),
        "consistently_good_rel":  sum(1 for p in per_prompt if p["halluc_rate_rel"]  <= 0.2),
        "avg_H_obj_missed": round(sum(det_missed_vals) / len(det_missed_vals), 4) if det_missed_vals else None,
        "avg_H_obj_extra":  round(sum(det_extra_vals)  / len(det_extra_vals),  4) if det_extra_vals  else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_json(per_prompt,    out_dir / "per_prompt_summary.json")
    _save_json(worst_prompts, out_dir / "worst_prompts.json")
    _save_json(best_prompts,  out_dir / "best_prompts.json")
    _save_json(summary,       out_dir / "experiment_3_summary.json")

    return summary


def run_experiment3():
    _header("Mode 3 — Experiment 3 (Per-prompt hallucination, 20 seeds)")
    print(f"\n  Runs       : {EXP3_N_RUNS}  (seeds {EXP3_BASE_SEED}–{EXP3_BASE_SEED + EXP3_N_RUNS - 1})")
    print(f"  Resolution : 1024×1024  (fixed)")
    print(f"  QA pairs reused from: {REGULAR['qa_pairs']}")

    if not REGULAR["qa_pairs"].exists():
        print(f"\n  WARNING: QA pairs not found at {REGULAR['qa_pairs']}")
        print("  Run Mode 0 steps 1-2 first.")
        if not _confirm("Continue anyway?"):
            return

    selected = _ask_steps(EXP3_STEPS)

    run_dirs = [EXP3_ROOT / f"run_{i+1:03d}" for i in range(EXP3_N_RUNS)]

    if "images" in selected:
        for i, run_dir in enumerate(run_dirs):
            seed = EXP3_BASE_SEED + i
            print(f"\n  Generating images: run {i+1:03d}/{EXP3_N_RUNS}  (seed={seed})")

            def _make_overrides(s):
                def _overrides(pid):
                    return {
                        EXP3_KSAMPLER_NODE:  {"seed": s},
                        EXP3_SAVEIMAGE_NODE: {"filename_prefix": pid},
                    }
                return _overrides

            step_images(
                workflow_path=REGULAR["workflow"],
                out_dir=run_dir / "images",
                extra_node_overrides=_make_overrides(seed),
            )

    for i, run_dir in enumerate(run_dirs):
        detection_dir = run_dir / "detection"
        halluc_dir    = run_dir / "hallucination"

        print(f"\n  ── run {i+1:03d}/{EXP3_N_RUNS} ──")

        if "detect"    in selected: step_detect(run_dir / "images", detection_dir)
        if "attrs"     in selected: step_attrs(detection_dir)
        if "relations" in selected: step_relations(detection_dir)
        if "halluc"    in selected: step_halluc(detection_dir, halluc_dir)

    if "aggregate" in selected:
        _section("Aggregating results across runs")
        summary = _aggregate_exp3(
            run_dirs=run_dirs,
            parsed_prompts_path=REGULAR["parsed"],
            out_dir=EXP3_ROOT / "aggregated",
            top_n=EXP3_TOP_N,
        )

        _header("Experiment 3 Results")
        print(f"  Prompts evaluated        : {summary['n_prompts']}")
        print(f"  Runs per prompt          : {summary['n_runs']}")
        print(f"  {'':30}  {'H_obj':>8}  {'H_attr':>8}  {'H_rel':>8}")
        print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}")
        print(f"  {'Avg halluc rate':<30}  {summary['avg_halluc_rate_obj']:>8.4f}  {summary['avg_halluc_rate_attr']:>8.4f}  {summary['avg_halluc_rate_rel']:>8.4f}")
        print(f"  {'Avg H (score)':<30}  {summary['avg_H_obj']:>8.4f}  {summary['avg_H_attr']:>8.4f}  {summary['avg_H_rel']:>8.4f}")
        print(f"  {'Consistently bad  (≥80%)':<30}  {summary['consistently_bad_obj']:>8}  {summary['consistently_bad_attr']:>8}  {summary['consistently_bad_rel']:>8}")
        print(f"  {'Consistently good (≤20%)':<30}  {summary['consistently_good_obj']:>8}  {summary['consistently_good_attr']:>8}  {summary['consistently_good_rel']:>8}")
        if summary.get("avg_H_obj_missed") is not None:
            print(f"\n  {'Detection-layer (avg across runs)':<30}  {'Value':>8}")
            print(f"  {'─'*30}  {'─'*8}")
            print(f"  {'H_missed (missed objects)':<30}  {summary['avg_H_obj_missed']:>8.4f}")
            print(f"  {'H_extra  (extra objects)':<30}  {summary['avg_H_obj_extra']:>8.4f}")
        print(f"\n  Outputs saved → {EXP3_ROOT}/")
        print(f"    aggregated/per_prompt_summary.json")
        print(f"    aggregated/best_prompts.json")
        print(f"    aggregated/worst_prompts.json")
        print(f"    aggregated/experiment_3_summary.json")
    _save_results()

# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

MODES = [
    (0, "Regular execution   — full pipeline, baseline"),
    (1, "Experiment 1        — 5 latent sizes, hallucination comparison"),
    (2, "Experiment 2        — steps & CFG sensitivity, hallucination comparison"),
    (3, "Experiment 3        — 20 seeds per prompt, per-prompt hallucination analysis"),
]

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    _header("T2I Hallucination Evaluation Pipeline")

    # --- Model selection ---
    print("\n  Select model:")
    for i, name in enumerate(AVAILABLE_MODELS, 1):
        print(f"    {i}. {name}")
    print("\n  Enter model number: ", end="")
    try:
        model_idx  = int(input().strip()) - 1
        model_name = AVAILABLE_MODELS[model_idx]
    except (ValueError, IndexError, EOFError):
        print("  Invalid choice. Exiting.")
        sys.exit(1)

    _set_model(model_name)
    print(f"\n  Model       : {model_name}")
    print(f"  Output root : {MODELS_ROOT / model_name}/")
    print(f"  Shared dir  : {SHARED_DIR}/")

    # --- Mode selection ---
    print()
    for num, desc in MODES:
        print(f"    {num}. {desc}")
    print()
    print("  Select mode: ", end="")

    try:
        choice = int(input().strip())
    except (ValueError, EOFError):
        print("  Invalid choice. Exiting.")
        sys.exit(1)

    if choice == 0:
        run_regular()
    elif choice == 1:
        run_experiment1()
    elif choice == 2:
        run_experiment2()
    elif choice == 3:
        run_experiment3()
    else:
        print(f"  Unknown mode: {choice}")
        sys.exit(1)


if __name__ == "__main__":
    main()
