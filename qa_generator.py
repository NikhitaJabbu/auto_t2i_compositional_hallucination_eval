"""
QA Generator: generates one YES/NO question per object, attribute, and spatial relation
from parsed prompt data using Ollama (llama3.1:8b). Falls back to rule-based templates if LLM fails.
"""

import json
import re
import subprocess
import time
import urllib.request
import urllib.error
from typing import List, Dict


# Ollama config
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BIN   = "/home/njabbu/ollama-bin/bin/ollama"

_MAX_WAIT_SECS = 30


def _ollama_running() -> bool:
    """Checks if Ollama is accepting connections."""
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _ensure_ollama() -> None:
    """Starts ollama serve if not running and waits until ready."""
    if _ollama_running():
        return

    print("  [ollama] Service not detected — starting `ollama serve` …")
    try:
        subprocess.Popen(
            [OLLAMA_BIN, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Ollama binary not found at {OLLAMA_BIN}. "
            "Update OLLAMA_BIN in qa_generator.py to the correct path."
        )

    deadline = time.time() + _MAX_WAIT_SECS
    while time.time() < deadline:
        time.sleep(1)
        if _ollama_running():
            print("  [ollama] Service is ready.")
            return

    raise RuntimeError(
        f"ollama serve did not become ready within {_MAX_WAIT_SECS}s. "
        "Please start it manually with: ollama serve"
    )


def _call_ollama(prompt_text: str, retries: int = 3) -> str:
    """Sends prompt to Ollama and returns response text. Retries on failure."""
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", "").strip()
        except urllib.error.URLError as e:
            print(f"  [ollama] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                if "Connection refused" in str(e):
                    _ensure_ollama()
                time.sleep(2)
    return ""


# ─── Article helper ───────────────────────────────────────────────────

_VOWELS = frozenset("aeiou")

def _article(word: str) -> str:
    """Returns 'an' if word starts with a vowel, else 'a'."""
    w = word.strip().lower()
    return "an" if w and w[0] in _VOWELS else "a"


# ─── Fallback question builder ────────────────────────────────────────

def _fallback_question(item_type: str, item_info: Dict) -> str:
    """Rule-based question builder — used only when Ollama fails."""
    if item_type == "object":
        surface = item_info.get("surface") or item_info.get("canonical") or "object"
        return f"Is there {_article(surface)} {surface}?"

    if item_type == "attribute":
        value     = item_info.get("attribute") or item_info.get("value") or ""
        obj       = item_info.get("object") or "object"
        attr_type = item_info.get("attr_type") or item_info.get("type") or ""
        if attr_type == "count":
            return f"Are there {value} {obj}s?"
        return f"Is the {obj} {value}?"

    if item_type == "relation":
        subject  = item_info.get("subject") or "subject"
        relation = item_info.get("relation") or "related to"
        obj      = item_info.get("object")
        if obj:
            return f"Is the {subject} {relation} the {obj}?"
        return f"Is the {subject} {relation}?"

    return "Is this present in the image?"


# ─── LLM prompt builders ─────────────────────────────────────────────

def _object_prompt(name: str) -> str:
    """Builds LLM prompt to generate a yes/no question for an object."""
    return (
        f"Generate a single simple yes/no question to verify whether a {name} "
        f"is visible in an image. "
        f"The question MUST contain the exact word '{name}'. "
        f"Keep it short and direct, like 'Is there a {name}?' "
        f"Return ONLY the question, nothing else."
    )


def _attribute_prompt(obj: str, value: str, attr_type: str) -> str:
    """Builds LLM prompt to generate a yes/no question for an attribute."""
    if attr_type == "count":
        return (
            f"Generate a single simple yes/no question to verify whether there are "
            f"exactly {value} {obj}(s) in the image. "
            f"The question MUST contain the exact word '{value}'. "
            f"Keep it short and direct, like 'Are there {value} {obj}s?' "
            f"Return ONLY the question, nothing else."
        )
    return (
        f"Generate a single simple yes/no question to verify whether the {obj} "
        f"in the image is {value}. "
        f"The question MUST contain the exact word '{value}'. "
        f"Keep it short and direct, like 'Is the {obj} {value}?' "
        f"Return ONLY the question, nothing else."
    )


def _relation_prompt(subject: str, relation: str, obj: str) -> str:
    """Builds LLM prompt to generate a yes/no question for a spatial relation."""
    return (
        f"Generate a single simple yes/no question to verify whether the {subject} "
        f"is {relation} the {obj} in the image. "
        f"The question MUST use the words '{subject}', '{relation}', and '{obj}'. "
        f"Keep it short and direct, like 'Is the {subject} {relation} the {obj}?' "
        f"Return ONLY the question, nothing else."
    )


# ─── Evidence extractor ───────────────────────────────────────────────

def _extract_evidence(prompt: str, surface_form: str, window: int = 4) -> str:
    """Extracts a short surrounding context window around a word in the prompt."""
    if not surface_form:
        return ""
    match = re.search(re.escape(surface_form), prompt, re.IGNORECASE)
    if not match:
        return surface_form
    start, end = match.start(), match.end()
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r'\S+', prompt)]
    match_idx = [i for i, (ts, te, _) in enumerate(tokens) if ts < end and te > start]
    if not match_idx:
        return surface_form
    first_idx, last_idx = match_idx[0], match_idx[-1]
    left  = max(0, first_idx - window)
    right = min(len(tokens) - 1, last_idx + window)
    phrase = " ".join(t for _, _, t in tokens[left : right + 1])
    if left > 0:
        phrase = "..." + phrase
    if right < len(tokens) - 1:
        phrase = phrase + "..."
    return phrase


def _clean_question(text: str) -> str:
    """Extracts first valid question line from LLM output — strips quotes and extra lines."""
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if line.endswith("?"):
            return line
    first = text.strip().splitlines()[0].strip().strip('"').strip("'") if text.strip() else ""
    return (first + "?") if first else ""


# ─── QA Generator ────────────────────────────────────────────────────

class QAGeneratorOllama:
    """Generates YES/NO questions using Ollama. Falls back to templates if LLM fails."""

    def __init__(self):
        _ensure_ollama()

    def _generate_question(self, item_type: str, item_info: Dict, llm_prompt: str) -> str:
        """Asks Ollama for a question — falls back to rule-based template if response is unusable."""
        raw = _call_ollama(llm_prompt)
        if raw:
            q = _clean_question(raw)
            if q:
                return q
        q = _fallback_question(item_type, item_info)
        return q

    def generate_object_questions(self, prompt: str, objects: List[Dict], start_count: int = 1) -> List[Dict]:
        """Generates one yes/no question per object using canonical name from the triple."""
        qa_pairs = []
        for i, obj in enumerate(objects):
            name      = obj.get("name", "")
            surface   = obj.get("surface", name)
            item_info = {"canonical": name, "surface": surface}
            # Use canonical name — not surface — so Llama produces a simple direct question
            question  = self._generate_question("object", item_info, _object_prompt(name))
            qa_pairs.append({
                "question_id": f"q{start_count + i:02d}",
                "question":    question,
                "answer":      "YES",
                "type":        "object",
                "evidence":    _extract_evidence(prompt, surface or name),
            })
        return qa_pairs

    def generate_attribute_questions(self, prompt: str, attributes: List[Dict], id_to_obj: Dict, start_count: int = 1) -> List[Dict]:
        """Generates one yes/no question per attribute."""
        qa_pairs = []
        for i, attr in enumerate(attributes):
            obj         = id_to_obj.get(attr.get("obj", ""), {})
            obj_name    = obj.get("name", attr.get("obj", ""))
            obj_surface = obj.get("surface", obj_name)
            value       = attr.get("value", "")
            attr_type   = attr.get("type", "")
            item_info   = {"attribute": value, "object": obj_name, "surface": obj_surface, "attr_type": attr_type}
            question    = self._generate_question("attribute", item_info, _attribute_prompt(obj_name, value, attr_type))
            qa_pairs.append({
                "question_id": f"q{start_count + i:02d}",
                "question":    question,
                "answer":      "YES",
                "type":        "attribute",
                "evidence":    _extract_evidence(prompt, obj_surface),
            })
        return qa_pairs

    def generate_relation_questions(self, prompt: str, relations: List[Dict], id_to_obj: Dict, start_count: int = 1) -> List[Dict]:
        """Generates one yes/no question per SPATIAL relation only. Non-spatial skipped."""
        qa_pairs = []
        q_idx = start_count
        for rel in relations:
            # Only generate questions for spatial relations
            if rel.get("type") != "spatial":
                continue
            subj_name = id_to_obj.get(rel.get("subject", ""), {}).get("name", rel.get("subject", ""))
            obj_name  = id_to_obj.get(rel.get("object",  ""), {}).get("name", rel.get("object",  ""))
            rel_label = rel.get("rel", "")
            item_info = {"subject": subj_name, "relation": rel_label, "object": obj_name, "rel_type": "spatial"}
            question  = self._generate_question("relation", item_info, _relation_prompt(subj_name, rel_label, obj_name))
            qa_pairs.append({
                "question_id": f"q{q_idx:02d}",
                "question":    question,
                "answer":      "YES",
                "type":        "relation",
                "evidence":    _extract_evidence(prompt, rel_label),
            })
            q_idx += 1
        return qa_pairs

    def generate_for_record(self, record: Dict) -> Dict:
        """Generates all QA pairs for one parsed prompt record — objects, attributes, relations."""
        prompt     = record["prompt"]
        record_id  = record["id"]
        objects    = record.get("objects", [])
        attributes = record.get("attributes", [])
        relations  = record.get("relations", [])

        id_to_obj      = {obj["id"]: obj for obj in objects}
        expected_total = len(objects) + len(attributes) + len(relations)


        all_qa_pairs   = []
        question_count = 1

        if objects:
            obj_qas = self.generate_object_questions(prompt, objects, question_count)
            all_qa_pairs.extend(obj_qas)
            question_count += len(obj_qas)

        if attributes:
            attr_qas = self.generate_attribute_questions(prompt, attributes, id_to_obj, question_count)
            all_qa_pairs.extend(attr_qas)
            question_count += len(attr_qas)

        if relations:
            rel_qas = self.generate_relation_questions(prompt, relations, id_to_obj, question_count)
            all_qa_pairs.extend(rel_qas)
            question_count += len(rel_qas)

        # Deduplicate by normalised question text, preserving order
        seen_questions: set = set()
        unique_qa_pairs: list = []
        for qa in all_qa_pairs:
            key = qa["question"].strip().lower().rstrip("?").strip()
            if key not in seen_questions:
                seen_questions.add(key)
                unique_qa_pairs.append(qa)

        for i, qa in enumerate(unique_qa_pairs, start=1):
            qa["question_id"] = f"q{i:02d}"

        total_generated = len(unique_qa_pairs)

        return {
            "id":             record_id,
            "prompt":         prompt,
            "qa_pairs":       unique_qa_pairs,
            "total_qas":      total_generated,
            "expected_total": expected_total,
            "category_counts": {
                "objects":    len([qa for qa in unique_qa_pairs if qa["type"] == "object"]),
                "attributes": len([qa for qa in unique_qa_pairs if qa["type"] == "attribute"]),
                "relations":  len([qa for qa in unique_qa_pairs if qa["type"] == "relation"]),
            },
        }


QAGenerator = QAGeneratorOllama


# ─── File processor ───────────────────────────────────────────────────

def process_jsonl_file(input_file: str, output_file: str) -> list:
    """Processes all parsed prompt records and saves QA pairs incrementally."""
    generator = QAGeneratorOllama()
    records: list = []

    print(f"Loading records from {input_file} ...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_num}: {e}")
                    raise

    print(f"{len(records)} records loaded — starting QA generation ...")

    results: list = []
    for idx, record in enumerate(records, 1):
        result = generator.generate_for_record(record)
        results.append(result)

        # Save after every record so progress is not lost on interruption
        with open(output_file, "w", encoding="utf-8") as out_f:
            for res in results:
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"Done — {len(results)} records saved → {output_file}")
    return results


process_jsonl_file_ollama = process_jsonl_file


if __name__ == "__main__":
    process_jsonl_file("Outputs/parsed_prompts.jsonl", "Outputs/prompts_qapairs.jsonl")
