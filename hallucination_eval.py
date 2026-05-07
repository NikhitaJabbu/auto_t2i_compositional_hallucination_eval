"""
hallucination_eval.py — Hallucination evaluation for T2I scene graphs.

Pipeline:
    prompt QA (ground truth)
    → scene-graph QA  (same questions answered from SG only)
    → answer comparison (expected vs predicted)
    → separate hallucination rates per category

Hallucination formulas
----------------------
  Object QA    : H_obj       = (G_o - C_o) / G_o
  Attribute    : H_attr      = (G_a - C_a) / G_a
  Relation     : H_rel       = (G_r - C_r) / G_r   (spatial relations only)
  Extra-object : H_obj_extra = E_o / T_o            (detection-based, separate)

Inputs:
    --qa_pairs        prompts_qapairs.jsonl   (ground-truth QA from prompt side)
    --scene_graph     scenegraph.json         (predicted scene graph from image)
    --detection_json  detection_results.json  (object detection for extra-object signal)
    --out_dir         output directory

Outputs:
    scenegraph_qa_answers.json      per-question prediction vs expected
    hallucination_per_prompt.json   per-prompt counts and rates
    hallucination_summary.json      dataset-level averages
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import requests as _requests

# ---------------------------------------------------------------------------
# Ollama / Llama config
# ---------------------------------------------------------------------------

_OLLAMA_URL     = "http://localhost:11434/api/generate"
_OLLAMA_BASE    = "http://localhost:11434"
_OLLAMA_MODEL   = "llama3.1:8b"
_OLLAMA_WORKERS = 5   # parallel prompt threads
_OLLAMA_BIN     = "/home/njabbu/ollama-bin/bin/ollama"
_OLLAMA_MAX_WAIT = 30


def _ensure_ollama() -> None:
    """Raise RuntimeError immediately if Ollama is not reachable."""
    try:
        _requests.get(_OLLAMA_BASE, timeout=3)
        return
    except Exception:
        pass

    import subprocess, time
    print("  [ollama] Service not detected — starting `ollama serve` …")
    try:
        subprocess.Popen(
            [_OLLAMA_BIN, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Ollama binary not found at {_OLLAMA_BIN}. "
            "Start Ollama manually with: ollama serve"
        )

    deadline = time.time() + _OLLAMA_MAX_WAIT
    while time.time() < deadline:
        time.sleep(1)
        try:
            _requests.get(_OLLAMA_BASE, timeout=2)
            print("  [ollama] Service is ready.")
            return
        except Exception:
            pass

    raise RuntimeError(
        f"Ollama did not become ready within {_OLLAMA_MAX_WAIT}s. "
        "Start it manually with: ollama serve"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Text / answer normalization
# ---------------------------------------------------------------------------

# Canonical yes/no synonyms
_YES = frozenset({"yes", "true", "present", "1", "correct", "exists",
                  "found", "yeah", "yep"})
_NO  = frozenset({"no", "false", "absent", "0", "incorrect", "missing",
                  "not found", "none", "not present", "doesnt exist",
                  "doesn't exist", "unknown"})


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip edge punctuation."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".,!?;:\"'")
    return s


def normalize_answer(ans: Any) -> str:
    """Map answer to a canonical string; collapse yes/no variants."""
    if ans is None:
        return "no"
    s = normalize_text(str(ans))
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    return s


def compare_answers(predicted: str, expected: str) -> bool:
    """True when normalized predicted == normalized expected."""
    return normalize_answer(predicted) == normalize_answer(expected)


# ---------------------------------------------------------------------------
# Schema helpers — safe extraction from variable schemas
# ---------------------------------------------------------------------------

def safe_get_prompt_id(record: dict) -> str:
    """Return prompt id regardless of whether key is 'id' or 'prompt_id'."""
    return str(record.get("prompt_id") or record.get("id") or "")


def extract_qa_items(record: dict) -> list:
    """Return the list of QA pairs from a qa_pairs.jsonl record."""
    for key in ("qa_pairs", "questions", "items", "qa"):
        val = record.get(key)
        if isinstance(val, list) and val:
            return val
    return []


# ---------------------------------------------------------------------------
# Scene graph extraction helpers
# ---------------------------------------------------------------------------

def extract_scenegraph_nodes(sg_entry: dict) -> list:
    return sg_entry.get("nodes") or sg_entry.get("objects") or []


def extract_scenegraph_edges(sg_entry: dict) -> list:
    return sg_entry.get("edges") or sg_entry.get("relations") or []


def _node_label(node: dict) -> str:
    return normalize_text(str(node.get("label") or node.get("name") or ""))


def _edge_label(edge: dict, side: str) -> str:
    return normalize_text(str(edge.get(side, "")))


def _edge_relations(edge: dict) -> list:
    """Return all non-null relation strings from an edge (spatial only)."""
    rels = []
    # Only spatial keys used — non_spatial not included.
    for key in ("spatial", "relation", "predicate"):
        v = edge.get(key)
        if v is not None and str(v).strip().lower() not in ("", "none", "null"):
            rels.append(normalize_text(str(v)))
    return rels


def _labels_match(a: str, b: str) -> bool:
    """True if two normalized labels are compatible (exact or substring)."""
    a, b = normalize_text(a), normalize_text(b)
    return bool(a and b and (a == b or a in b or b in a))


def _word_present(word: str, text: str) -> bool:
    """True if `word` appears as a whole word in `text`."""
    if not word:
        return False
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


# ---------------------------------------------------------------------------
# Relation compatibility
# ---------------------------------------------------------------------------

# Maps SG spatial values → pure spatial phrasings ONLY.
_SPATIAL_COMPAT: dict = {
    "on":       {"on", "on top of", "atop"},
    "above":    {"above", "over"},
    "below":    {"below", "underneath", "under"},
    "under":    {"under", "underneath", "below"},
    "near":     {"near", "next to", "beside", "close to", "on the side of",
                 "adjacent to", "by"},
    "next to":  {"next to", "beside", "near", "adjacent to",
                 "on the side of", "by"},
    "inside":   {"inside", "within", "in", "contained in"},
    "contains": {"containing", "contains", "enclosing"},
    "left of":  {"left", "to the left of", "on the left", "left of"},
    "right of": {"right", "to the right of", "on the right", "right of"},
}



def _spatial_matches(sg_spatial: str, question: str) -> bool:
    """True if the SG spatial value is semantically present in the question."""
    if not sg_spatial:
        return False
    sg_spatial = normalize_text(sg_spatial)
    q = normalize_text(question)
    if _word_present(sg_spatial, q) or sg_spatial in q:
        return True
    for spatial_val, compat_set in _SPATIAL_COMPAT.items():
        if sg_spatial == spatial_val:
            for phrase in compat_set:
                if phrase in q:
                    return True
    return False




# ---------------------------------------------------------------------------
# Scene-graph QA engine
# ---------------------------------------------------------------------------

def answer_object_question(question: str, nodes: list) -> str:
    """
    Answer "Is there a/an X?" from scene graph nodes.
    Returns "yes" if any node label is mentioned in the question, else "no".
    Does NOT use the expected answer.
    """
    q = normalize_text(question)
    for node in nodes:
        lbl = _node_label(node)
        if lbl and _word_present(lbl, q):
            return "yes"
    return "no"


def answer_attribute_question(question: str, nodes: list) -> str:
    q = normalize_text(question)
    for node in nodes:
        lbl = _node_label(node)
        if not lbl or not _word_present(lbl, q):
            continue
        # Node is mentioned in the question — now check its attributes
        attrs = node.get("attributes") or {}
        for attr_key, attr_val in attrs.items():
            if attr_val is None:
                continue
            val_norm = normalize_text(str(attr_val))
            if val_norm and _word_present(val_norm, q):
                return "yes"
        # The node exists in SG but none of its attribute values match
        return "no"
    # Node not found in SG
    return "no"


def answer_relation_question(question: str, nodes: list, edges: list) -> str:
    """
    Answer "Is the X [rel] the Y?" from scene graph edges.

    Logic:
      1. Collect all SG node labels mentioned in the question.
      2. For each edge that connects two of those mentioned nodes,
         check whether any of the edge's spatial relations
         are semantically compatible with the question text.
      3. If a matching edge is found → "yes"; otherwise → "no".
    """
    q = normalize_text(question)

    # Collect nodes whose labels appear in the question
    mentioned = [n for n in nodes
                 if _node_label(n) and _word_present(_node_label(n), q)]

    if len(mentioned) < 2:
        return "no"

    mentioned_labels = {_node_label(n) for n in mentioned}

    for edge in edges:
        from_lbl = _edge_label(edge, "from")
        to_lbl   = _edge_label(edge, "to")

        # Edge must connect two of the mentioned nodes (either direction)
        from_ok = any(_labels_match(from_lbl, ml) for ml in mentioned_labels)
        to_ok   = any(_labels_match(to_lbl,   ml) for ml in mentioned_labels)
        if not (from_ok and to_ok):
            continue

        # Check if any edge relation is semantically present in the question
        for rel in _edge_relations(edge):
            if _spatial_matches(rel, q):
                return "yes"

        # Edge exists between the right pair — if question asks about existence
        # of ANY relation (no specific relation phrase detectable), return "yes"
        # This handles questions that just ask "Is X related to Y?" generically.
        if not _edge_relations(edge):
            return "yes"

    return "no"


def answer_question_from_scene_graph(item: dict, sg_entry: dict) -> str:
    """
    Master dispatcher: answer one QA item using ONLY scene graph data.

    Does NOT read the expected answer from `item["answer"]`.
    Does NOT use prompt text for inference.

    Routes by item["type"]: "object" | "attribute" | "relation"
    """
    nodes    = extract_scenegraph_nodes(sg_entry)
    edges    = extract_scenegraph_edges(sg_entry)
    question = item.get("question", "")
    q_type   = normalize_text(str(item.get("type", "object")))

    if q_type == "object":
        return answer_object_question(question, nodes)
    if q_type == "attribute":
        return answer_attribute_question(question, nodes)
    if q_type == "relation":
        return answer_relation_question(question, nodes, edges)
    return "no"


# ---------------------------------------------------------------------------
# Required-objects extraction (for extra-object detection)
# ---------------------------------------------------------------------------

def extract_required_objects_from_qa(qa_items: list) -> set:
    """
    Collect the set of required object label strings from all QA items.
    These are the objects the prompt demands; any detected object outside
    this set counts as extra (hallucinated).

    Extraction rules:
      object   "Is there a X?"          → last noun token before "?"
      attribute "Is the X Y?"           → first noun after "Is the"
      relation  "Is the X [rel] the Y?" → first noun (subject) and
                                          last noun (object)
    """
    required: set = set()

    for item in qa_items:
        q      = normalize_text(item.get("question", ""))
        q_type = normalize_text(str(item.get("type", "object")))

        if q_type == "object":
            # "Is there a/an [adj*] NOUN"
            m = re.match(r"is there an?\s+(?:\w+\s+)*?(\w+)\s*$", q)
            if m:
                required.add(m.group(1))
            else:
                # "Are there N NOUNs"
                m2 = re.match(r"are there\s+(?:\w+\s+)*?(\w+?)s?\s*$", q)
                if m2:
                    required.add(m2.group(1))

        elif q_type == "attribute":
            # "Is the NOUN adjective"
            m = re.match(r"is the\s+(\w+)", q)
            if m:
                required.add(m.group(1))

        elif q_type == "relation":
            # "Is the SUBJ ... the OBJ"
            m = re.match(r"is the\s+(\w+).+\bthe\s+(\w+)\s*$", q)
            if m:
                required.add(m.group(1))
                required.add(m.group(2))
            else:
                # No trailing "the OBJ" — take first noun after "Is the"
                m2 = re.match(r"is the\s+(\w+)", q)
                if m2:
                    required.add(m2.group(1))

    return required


# ---------------------------------------------------------------------------
# Extra-object detection from detection_results.json
# ---------------------------------------------------------------------------

def extract_detected_object_labels(det_entry: dict) -> list:
    """
    Return a flat list of detected object labels from a detection entry.
    Tries 'objects_detected' then 'correct'; also reads pre-computed 'extra'.
    """
    labels: list = []

    # Pre-computed extra field (objects already flagged as non-required)
    for o in det_entry.get("extra", []) or []:
        if isinstance(o, dict):
            lbl = normalize_text(str(o.get("label") or o.get("name") or ""))
            if lbl:
                labels.append(lbl)
        elif isinstance(o, str):
            labels.append(normalize_text(o))

    # If 'extra' is populated we have our answer; return early
    if labels:
        return labels

    # Fallback: derive from full objects_detected list
    for key in ("objects_detected", "correct", "detected", "objects"):
        objs = det_entry.get(key)
        if not isinstance(objs, list):
            continue
        for o in objs:
            if isinstance(o, dict):
                lbl = normalize_text(str(o.get("label") or o.get("name") or ""))
                cnt = int(o.get("count", 1) or 1)
                if lbl:
                    labels.extend([lbl] * cnt)
            elif isinstance(o, str):
                labels.append(normalize_text(o))
        break  # only use the first matching key

    return labels


def count_extra_objects(det_entry: dict, required_labels: set) -> int:
    """
    Count detected objects NOT covered by the required set from QA.

    If the detection entry already has a pre-computed 'extra' list, use its
    length directly. Otherwise compute from all detected labels vs required set.
    """
    pre_extra = [o for o in (det_entry.get("extra") or [])
                 if o is not None and (isinstance(o, dict) and
                    (o.get("label") or o.get("name"))) or isinstance(o, str)]
    if pre_extra:
        return len(pre_extra)

    detected = extract_detected_object_labels(det_entry)
    extras = 0
    for lbl in detected:
        if not any(_labels_match(lbl, req) for req in required_labels):
            extras += 1
    return extras


def count_total_detected_objects(det_entry: dict) -> int:
    """
    Count the total number of detected objects from the detection entry,
    used as the denominator in the extra-object hallucination formula:
        H_obj_extra = E_o / T_o
    """
    for key in ("objects_detected", "correct", "detected", "objects"):
        objs = det_entry.get(key)
        if not isinstance(objs, list):
            continue
        total = 0
        for o in objs:
            if isinstance(o, dict):
                total += int(o.get("count", 1) or 1)
            elif isinstance(o, str) and o.strip():
                total += 1
        return total
    return 0


# ---------------------------------------------------------------------------
# Per-prompt scoring
# ---------------------------------------------------------------------------

def _referenced_node_labels(question: str, nodes: list) -> list:
    """Return all node labels that appear as whole words in the question."""
    q = normalize_text(question)
    return [_node_label(n) for n in nodes
            if _node_label(n) and _word_present(_node_label(n), q)]


# ---------------------------------------------------------------------------
# Scene-graph serialisation and Llama batch answering
# ---------------------------------------------------------------------------

def _sg_to_text(nodes: list, edges: list) -> str:
    """
    Serialize a scene graph to a short, human-readable string suitable for
    pasting into a Llama prompt.

    Example output:
        Objects: chair (color: brown; material: wood); table (color: white)
        Relations: chair is left of table; vase is on table
    """
    parts: list = []

    if nodes:
        obj_strs: list = []
        for n in nodes:
            lbl = _node_label(n)
            if not lbl:
                continue
            details: list = []
            cnt = n.get("count")
            if cnt is not None and str(cnt).strip() not in ("", "1"):
                details.append(f"count: {cnt}")
            for k, v in (n.get("attributes") or {}).items():
                if v is not None:
                    details.append(f"{k}: {v}")
            obj_strs.append(lbl + (f" ({'; '.join(details)})" if details else ""))
        parts.append("Objects: " + ("; ".join(obj_strs) if obj_strs else "none"))
    else:
        parts.append("Objects: none detected")

    if edges:
        rel_strs: list = []
        for e in edges:
            from_lbl = _edge_label(e, "from")
            to_lbl   = _edge_label(e, "to")
            for rel in _edge_relations(e):
                rel_strs.append(f"{from_lbl} is {rel} {to_lbl}")
        if rel_strs:
            parts.append("Relations: " + "; ".join(rel_strs))

    return "\n".join(parts)


def _llama_answer_batch(sg_text: str, questions: list) -> dict:
    """
    Send all questions for one prompt to Llama in a single Ollama call.

    Parameters
    ----------
    sg_text   : serialised scene graph (from _sg_to_text)
    questions : list of (qid, question_text, evidence) tuples
                  evidence — the prompt snippet that generated the question

    Returns
    -------
    dict mapping qid → "yes" | "no"
    Defaults to "no" for any question whose answer cannot be parsed.
    """
    if not questions:
        return {}

    # Build numbered question list; include evidence snippet as context
    q_lines: list = []
    for i, (qid, question, evidence) in enumerate(questions, 1):
        if evidence:
            q_lines.append(f"Q{i} (context: \"{evidence}\"): {question}")
        else:
            q_lines.append(f"Q{i}: {question}")

    prompt = (
        "You are a precise scene-graph evaluator.\n"
        "Answer each question using ONLY the scene graph provided below.\n\n"
        f"Scene Graph:\n{sg_text}\n\n"
        "Rules:\n"
        "- Answer ONLY 'yes' or 'no' for each question.\n"
        "- Base your answer SOLELY on what is explicitly stated in the scene graph.\n"
        "- If the scene graph does not explicitly confirm something, answer 'no'.\n"
        "- Do NOT use world knowledge or make assumptions.\n\n"
        + "\n".join(q_lines)
        + "\n\nRespond with one answer per line, exactly as:\n"
        "Q1: yes\nQ2: no\n(etc.)"
    )

    try:
        resp = _requests.post(
            _OLLAMA_URL,
            json={
                "model":   _OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=90,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")
    except Exception as exc:
        print(f"    [Llama ERROR] {exc} — defaulting all to 'no'")
        return {qid: "no" for qid, _, _ in questions}

    # Parse "Q1: yes" / "Q1: no" patterns (case-insensitive)
    answers: dict = {}
    for i, (qid, _, _) in enumerate(questions, 1):
        m = re.search(rf"Q{i}\s*:\s*(yes|no)", response_text, re.IGNORECASE)
        answers[qid] = m.group(1).lower() if m else "no"

    return answers


def score_prompt(
    pid: str,
    qa_items: list,
    sg_entry: dict,
    det_entry: dict,
    is_single_detected: bool = False,
    collision_dropped_labels: set = None,
) -> tuple:
    """
    Score hallucination for one prompt using Llama + hard DSG-style cascade.

    Collision prompts are included; the dropped object's question is forced "no".

    Pass 1 — Hard existence check + collect Llama questions
    -------------------------------------------------------
    For each QA item:
      • sg_empty                            → cascade_fail (no Llama call)
      • relation + is_single_detected       → excluded_single_detected
      • attr/rel: ref object not in SG      → cascade_fail (parent object miss)
      • everything else                     → queued for Llama

    All Llama-destined questions for this prompt are sent in ONE batch call
    so Llama has full scene-graph context for all questions simultaneously.

    Pass 2 — Score
    --------------
    predicted answer (hard or Llama) vs expected answer → matched bool
    Accumulate G_o/C_o, G_a/C_a, G_r/C_r per DSG rules.

    Formulas
    --------
    H_obj   = (G_o - C_o) / G_o          all non-collision prompts
    H_attr  = (G_a - C_a) / G_a          all non-collision prompts
    H_rel   = (G_r - C_r) / G_r          None if is_single_detected
    H_obj_extra = E_o / T_o              detection-based, separate

    Returns
    -------
    scores  : dict  per-prompt counts, rates, and flags
    qa_recs : list  per-question comparison records
    """
    nodes = extract_scenegraph_nodes(sg_entry)
    edges = extract_scenegraph_edges(sg_entry)

    found_objects: set = {_node_label(n) for n in nodes if _node_label(n)}
    sg_empty = len(nodes) == 0

    sg_text = _sg_to_text(nodes, edges)

    # ── Pass 1: Classify each question ────────────────────────────────────
    # hard_answers: qid → (predicted_str, is_cascade: bool, is_excluded: bool)
    hard_answers: dict = {}
    llama_questions: list = []   # (qid, question_text, evidence_str)

    for item in qa_items:
        qid      = str(item.get("question_id", ""))
        q_type   = normalize_text(str(item.get("type", "object")))
        question = item.get("question", "")
        evidence = item.get("evidence", "")

        # Any question referencing a dropped collision object → predict no
        if collision_dropped_labels:
            if any(normalize_text(lbl) in normalize_text(question) for lbl in collision_dropped_labels):
                hard_answers[qid] = ("no", False, False)
                continue

        # Single-detected: second object missing so relation cannot hold, predict no
        if q_type == "relation" and is_single_detected:
            hard_answers[qid] = ("no", False, False)
            continue

        # SG completely empty → every question is a cascade fail
        if sg_empty:
            hard_answers[qid] = ("cascade_fail", True, False)
            continue

        # Attr / rel: if the referenced parent object is not in the SG,
        # cascade (don't call Llama — it can't answer either)
        if q_type in ("attribute", "relation"):
            ref_labels = _referenced_node_labels(question, nodes)
            if ref_labels and any(lbl not in found_objects for lbl in ref_labels):
                hard_answers[qid] = ("cascade_fail", True, False)
                continue

        # Passes hard check → queue for Llama
        llama_questions.append((qid, question, evidence))

    # ── Single Llama batch call for this prompt ───────────────────────────
    llama_answers = _llama_answer_batch(sg_text, llama_questions) if llama_questions else {}

    # ── Pass 2: Score all questions ───────────────────────────────────────
    G_o = C_o = 0
    G_a = C_a = G_a_cascade = 0
    G_r = C_r = G_r_cascade = 0
    qa_recs: list = []

    for item in qa_items:
        qid      = str(item.get("question_id", ""))
        q_type   = normalize_text(str(item.get("type", "object")))
        expected = normalize_answer(item.get("answer", "yes"))
        question = item.get("question", "")

        if qid in hard_answers:
            predicted, is_cascade, is_excluded = hard_answers[qid]
        else:
            predicted   = llama_answers.get(qid, "no")
            is_cascade  = False
            is_excluded = False

        if is_excluded:
            qa_recs.append({
                "question_id":      qid,
                "question":         question,
                "type":             q_type,
                "expected_answer":  expected,
                "predicted_answer": predicted,
                "match":            False,
                "cascade_fail":     False,
                "excluded":         True,
            })
            continue  # do not count in any G counter

        matched = compare_answers(predicted, expected) if not is_cascade else False

        if q_type == "object":
            G_o += 1
            if matched:
                C_o += 1
        elif q_type == "attribute":
            G_a += 1
            if matched:
                C_a += 1
            if is_cascade:
                G_a_cascade += 1
        elif q_type == "relation":
            G_r += 1
            if matched:
                C_r += 1
            if is_cascade:
                G_r_cascade += 1

        qa_recs.append({
            "question_id":      qid,
            "question":         question,
            "type":             q_type,
            "expected_answer":  expected,
            "predicted_answer": predicted,
            "match":            matched,
            "cascade_fail":     is_cascade,
        })

    # ── Hallucination rates ────────────────────────────────────────────────
    H_obj  = (G_o - C_o) / G_o if G_o > 0 else 0.0
    H_attr = (G_a - C_a) / G_a if G_a > 0 else 0.0
    H_rel  = (G_r - C_r) / G_r if G_r > 0 else 0.0

    scores = {
        "prompt_id":          pid,
        "is_single_detected": is_single_detected,
        "G_o":   G_o,
        "C_o":   C_o,
        "H_obj": round(H_obj, 4),
        "G_a":   G_a,
        "C_a":   C_a,
        "H_attr": round(H_attr, 4),
        "G_r":   G_r,
        "C_r":   C_r,
        "H_rel": round(H_rel, 4),
    }
    return scores, qa_recs


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def compute_dataset_summary(all_scores: list) -> dict:
    """
    Compute dataset-level hallucination averages.

    H_obj  : averaged over all N evaluated prompts
    H_attr : averaged over all N evaluated prompts
    H_rel  : averaged over prompts that have at least one relation question (G_r > 0)
    """
    N = len(all_scores)
    if N == 0:
        return {"num_prompts": 0}

    rel_valid = [s for s in all_scores if s["G_r"] > 0]
    N_rel     = len(rel_valid)

    avg_H_obj  = sum(s["H_obj"]  for s in all_scores) / N
    avg_H_attr = sum(s["H_attr"] for s in all_scores) / N
    avg_H_rel  = (sum(s["H_rel"] for s in rel_valid) / N_rel) if N_rel else 0.0

    return {
        "num_prompts":     N,
        "num_rel_prompts": N_rel,
        "avg_H_obj":       round(avg_H_obj,  4),
        "avg_H_attr":      round(avg_H_attr, 4),
        "avg_H_rel":       round(avg_H_rel,  4),
    }


# ---------------------------------------------------------------------------
# Parallel worker wrapper
# ---------------------------------------------------------------------------

def _score_one(args: tuple) -> tuple:
    """Thin wrapper so score_prompt() can run in a ThreadPoolExecutor."""
    pid, qa_items, sg_entry, det_entry, is_single_detected, prompt_text, collision_dropped_labels = args
    scores, qa_recs = score_prompt(pid, qa_items, sg_entry, det_entry, is_single_detected, collision_dropped_labels)
    return scores, qa_recs, pid, is_single_detected, prompt_text


# ---------------------------------------------------------------------------
# Core evaluation — callable from run.py or CLI
# ---------------------------------------------------------------------------

def run_evaluation(
    qa_pairs_path: str,
    scene_graph_path: str,
    detection_json_path: str,
    out_dir: str,
) -> dict:
    """
    Run the full hallucination evaluation pipeline.

    Parameters
    ----------
    qa_pairs_path       : path to prompts_qapairs.jsonl
    scene_graph_path    : path to scenegraph.json
    detection_json_path : path to detection_results.json
    out_dir             : directory to write outputs

    Returns
    -------
    summary dict  (avg_H_obj, avg_H_attr, avg_H_rel, num_prompts, num_rel_prompts)
    """
    _ensure_ollama()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---
    qa_raw   = load_jsonl(Path(qa_pairs_path))
    sg_list  = load_json(Path(scene_graph_path))
    det_list = load_json(Path(detection_json_path))

    # Index by prompt_id for O(1) lookup
    sg_by_pid  = {safe_get_prompt_id(e): e for e in sg_list}
    det_by_pid = {safe_get_prompt_id(e): e for e in det_list}

    # Collision prompts: dropped object gets forced "no"; winner is scored normally
    collision_dropped_by_pid: dict = {
        safe_get_prompt_id(e): {d["label"] for d in e.get("collision_dropped", [])}
        for e in det_list
        if e.get("collision_dropped")
    }
    n_collision_prompts = len(collision_dropped_by_pid)

    # Single-detected: one object found, one missed — relation answered "no"
    single_detected_pids: set = {
        safe_get_prompt_id(e) for e in det_list
        if len(e.get("missing", [])) > 0
    }

    # ── Build task list ───────────────────────────────────────────────────
    tasks: list = []
    for record in qa_raw:
        pid      = safe_get_prompt_id(record)
        qa_items = extract_qa_items(record)

        if not qa_items:
            continue

        sg_entry                 = sg_by_pid.get(pid)  or {}
        det_entry                = det_by_pid.get(pid) or {}
        is_single_detected       = pid in single_detected_pids
        collision_dropped_labels = collision_dropped_by_pid.get(pid, set())

        tasks.append((
            pid, qa_items, sg_entry, det_entry,
            is_single_detected, record.get("prompt", ""),
            collision_dropped_labels,
        ))

    all_scores:    list = [None] * len(tasks)
    all_qa_output: list = [None] * len(tasks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_OLLAMA_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_score_one, t): i for i, t in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                scores, qa_recs, pid, is_sd, prompt_text = future.result()
            except Exception:
                pid, qa_items_i, _, _, is_sd, prompt_text, _ = tasks[i]
                g_o = sum(1 for q in qa_items_i if normalize_text(str(q.get("type",""))) == "object")
                g_a = sum(1 for q in qa_items_i if normalize_text(str(q.get("type",""))) == "attribute")
                g_r = sum(1 for q in qa_items_i if normalize_text(str(q.get("type",""))) == "relation")
                all_scores[i] = {
                    "prompt_id": pid, "is_single_detected": is_sd,
                    "G_o": g_o, "C_o": 0, "H_obj":  1.0 if g_o > 0 else 0.0,
                    "G_a": g_a, "C_a": 0, "H_attr": 1.0 if g_a > 0 else 0.0,
                    "G_r": g_r, "C_r": 0, "H_rel":  1.0 if g_r > 0 else 0.0,
                    "scoring_failed": True,
                }
                all_qa_output[i] = {
                    "prompt_id": pid, "prompt": prompt_text,
                    "is_single_detected": is_sd, "qa_comparisons": [],
                    "scoring_failed": True,
                }
                continue
            all_scores[i] = scores
            all_qa_output[i] = {
                "prompt_id":          pid,
                "prompt":             prompt_text,
                "is_single_detected": is_sd,
                "qa_comparisons":     qa_recs,
            }

    # Filter out any unfilled slots (failed before task was submitted)
    all_scores    = [s for s in all_scores    if s is not None]
    all_qa_output = [s for s in all_qa_output if s is not None]

    # --- Compute summary ---
    summary = compute_dataset_summary(all_scores)

    # --- Save outputs ---
    save_json(all_qa_output, out_dir / "scenegraph_qa_answers.json")
    save_json(all_scores,    out_dir / "hallucination_per_prompt.json")
    save_json(summary,       out_dir / "hallucination_summary.json")

    # --- Console report ---
    w = 66
    s = summary
    print(f"\n{'='*w}")
    print(f"  Hallucination Evaluation Summary")
    print(f"{'─'*w}")
    print(f"  Total prompts                              : {len(qa_raw)}")
    print(f"  Collision prompts (winner kept)            : {n_collision_prompts}")
    print(f"  Prompts evaluated                          : {s['num_prompts']}")
    print(f"  Object hallucination      (G_o-C_o)/G_o   : {s['avg_H_obj']:.4f}")
    print(f"{'─'*w}")
    print(f"  Attribute hallucination   (G_a-C_a)/G_a   : {s['avg_H_attr']:.4f}  over {s['num_prompts']} prompts")
    print(f"{'─'*w}")
    print(f"  Relation hallucination    (G_r-C_r)/G_r   : {s['avg_H_rel']:.4f}  over {s['num_rel_prompts']} prompts")
    print(f"{'='*w}\n")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hallucination evaluation via scene-graph QA comparison.")
    parser.add_argument("--qa_pairs",
                        default="Outputs/prompts_qapairs.jsonl")
    parser.add_argument("--scene_graph",
                        default="Outputs/detection_results/scenegraph.json")
    parser.add_argument("--detection_json",
                        default="Outputs/detection_results/detection_results.json")
    parser.add_argument("--out_dir",
                        default="Outputs/hallucination")
    args = parser.parse_args()

    run_evaluation(args.qa_pairs, args.scene_graph, args.detection_json, args.out_dir)


if __name__ == "__main__":
    main()