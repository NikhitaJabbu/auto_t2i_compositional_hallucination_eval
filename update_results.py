"""
update_results.py — Regenerate RESULTS.md from all available pipeline outputs.

Always rewrites the entire file so results are never stale.
Organised by model: each model gets its own section, experiments appear as
subsections only when data exists for that model.
Reads from Outputs/models/{model}/ and writes RESULTS.md in the project root.
"""

import json
from datetime import datetime
from pathlib import Path

MODELS_ROOT = Path("Outputs/models")
OUT_FILE    = Path("RESULTS.md")

DISPLAY_NAME = {
    "flux_schnell": "Flux Schnell",
    "sd15":         "SD 1.5",
    "sd35":         "SD 3.5",
    "sdxl":         "SDXL",
}


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _sep(*widths):
    return "| " + " | ".join("-" * max(w, 3) for w in widths) + " |"


# ---------------------------------------------------------------------------
# Section builders (no model column — model is the heading)
# ---------------------------------------------------------------------------

def _section_regular(model):
    hs = _load(MODELS_ROOT / model / "baseline" / "hallucination" / "hallucination_summary.json")
    ds = _load(MODELS_ROOT / model / "baseline" / "detection"     / "detection_summary.json")
    ls = _load(MODELS_ROOT / model / "baseline" / "latency_summary.json")
    if hs is None:
        return []

    header = _row("H_obj", "H_attr", "H_rel", "H_missed", "H_extra", "#Prompts")
    sep    = _sep(8, 8, 8, 10, 8, 10)
    data   = _row(
        _fmt(hs.get("avg_H_obj")),
        _fmt(hs.get("avg_H_attr")),
        _fmt(hs.get("avg_H_rel")),
        _fmt(ds.get("H_obj_missed") if ds else None),
        _fmt(ds.get("H_obj_extra")  if ds else None),
        hs.get("num_prompts", "—"),
    )
    lines = ["### Regular Baseline", "", header, sep, data, ""]

    if ls:
        steps = ls.get("steps", {})
        lat_header = _row("Parse(s)", "QA Gen(s)", "Img Gen(s)", "Detection(s)",
                          "Attr Pred(s)", "Relations(s)", "Halluc Eval(s)", "Total(s)", "ImgGen/p(s)")
        lat_sep    = _sep(10, 10, 11, 13, 13, 13, 14, 10, 11)
        lat_data   = _row(
            _fmt(steps.get("parse"),     decimals=1),
            _fmt(steps.get("qa"),        decimals=1),
            _fmt(steps.get("images"),    decimals=1),
            _fmt(steps.get("detect"),    decimals=1),
            _fmt(steps.get("attrs"),     decimals=1),
            _fmt(steps.get("relations"), decimals=1),
            _fmt(steps.get("halluc"),    decimals=1),
            _fmt(ls.get("total_s"),              decimals=1),
            _fmt(ls.get("img_gen_per_prompt_s"), decimals=2),
        )
        lines += ["**Latency (wall-clock)**", "", lat_header, lat_sep, lat_data, ""]

    return lines


def _section_exp1(model):
    data = _load(MODELS_ROOT / model / "experiment_1" / "experiment_1_comparison.json")
    if not data:
        return []

    header = _row("Size", "H_obj", "H_attr", "H_rel", "H_missed", "H_extra", "#Prompts", "#Rel")
    sep    = _sep(12, 8, 8, 8, 10, 8, 10, 6)
    rows   = [
        _row(
            s.get("latent_size", "—"),
            _fmt(s.get("avg_H_obj")),
            _fmt(s.get("avg_H_attr")),
            _fmt(s.get("avg_H_rel")),
            _fmt(s.get("H_obj_missed")),
            _fmt(s.get("H_obj_extra")),
            s.get("num_prompts", "—"),
            s.get("num_rel_prompts", "—"),
        )
        for s in data
    ]
    return ["### Experiment 1 — Latent Sizes", "", header, sep] + rows + [""]


def _section_exp2(model):
    data = _load(MODELS_ROOT / model / "experiment_2" / "experiment_2_comparison.json")
    if not data:
        return []

    header = _row("Tag", "Steps", "CFG", "H_obj", "H_attr", "H_rel", "H_missed", "H_extra", "#Prompts")
    sep    = _sep(12, 6, 6, 8, 8, 8, 10, 8, 10)

    lines = ["### Experiment 2 — Steps & CFG Sensitivity", ""]

    steps = data.get("steps_sweep", [])
    if steps:
        lines += ["#### Steps Sweep (CFG fixed at 8)", "", header, sep]
        lines += [
            _row(
                s.get("tag", "—"), s.get("steps", "—"),
                _fmt(s.get("cfg"), decimals=1),
                _fmt(s.get("avg_H_obj")), _fmt(s.get("avg_H_attr")), _fmt(s.get("avg_H_rel")),
                _fmt(s.get("H_obj_missed")), _fmt(s.get("H_obj_extra")),
                s.get("num_prompts", "—"),
            )
            for s in steps
        ]
        lines.append("")

    cfg = data.get("cfg_sweep", [])
    if cfg:
        lines += ["#### CFG Sweep (steps fixed at 20)", "", header, sep]
        lines += [
            _row(
                s.get("tag", "—"), s.get("steps", "—"),
                _fmt(s.get("cfg"), decimals=1),
                _fmt(s.get("avg_H_obj")), _fmt(s.get("avg_H_attr")), _fmt(s.get("avg_H_rel")),
                _fmt(s.get("H_obj_missed")), _fmt(s.get("H_obj_extra")),
                s.get("num_prompts", "—"),
            )
            for s in cfg
        ]
        lines.append("")

    return lines if len(lines) > 2 else []


def _section_exp3(model):
    s = _load(MODELS_ROOT / model / "experiment_3" / "aggregated" / "experiment_3_summary.json")
    if not s:
        return []

    header = _row("#Prompts", "#Runs",
                  "H_obj", "H_attr", "H_rel",
                  "Rate_obj", "Rate_attr", "Rate_rel",
                  "BadObj(≥80%)", "GoodObj(≤20%)",
                  "H_missed", "H_extra")
    sep  = _sep(10, 7, 8, 8, 8, 10, 10, 10, 12, 13, 10, 8)
    data = _row(
        s.get("n_prompts", "—"),
        s.get("n_runs",    "—"),
        _fmt(s.get("avg_H_obj")),
        _fmt(s.get("avg_H_attr")),
        _fmt(s.get("avg_H_rel")),
        _fmt(s.get("avg_halluc_rate_obj")),
        _fmt(s.get("avg_halluc_rate_attr")),
        _fmt(s.get("avg_halluc_rate_rel")),
        s.get("consistently_bad_obj",  "—"),
        s.get("consistently_good_obj", "—"),
        _fmt(s.get("avg_H_obj_missed")),
        _fmt(s.get("avg_H_obj_extra")),
    )
    return ["### Experiment 3 — Seed Variance (20 runs)", "", header, sep, data, ""]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models = sorted(
        d.name for d in MODELS_ROOT.iterdir() if d.is_dir()
    ) if MODELS_ROOT.exists() else []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = [
        "# Pipeline Results",
        "",
        f"*Last updated: {timestamp}*",
        "",
        "---",
        "",
    ]

    active = []
    for model in models:
        sections = (
            _section_regular(model)
            + _section_exp1(model)
            + _section_exp2(model)
            + _section_exp3(model)
        )
        if sections:
            active.append((model, sections))

    for i, (model, sections) in enumerate(active):
        display = DISPLAY_NAME.get(model, model.upper())
        doc.append(f"## {display}")
        doc.append("")
        doc.extend(sections)
        if i < len(active) - 1:
            doc += ["---", ""]

    OUT_FILE.write_text("\n".join(doc), encoding="utf-8")
    print(f"  Updated → {OUT_FILE}  ({len(models)} model(s): {', '.join(models) or 'none'})")


if __name__ == "__main__":
    main()
