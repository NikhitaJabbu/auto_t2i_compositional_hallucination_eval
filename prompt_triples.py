"""
This is the initial stage of the system. 
From each prompt objects, attributes (color/shape/material/count), and spatial relations are extracted using SpaCy.
"""

import json
from pathlib import Path
import spacy

# Input and output file paths
INPUT_PATH  = Path("data/promts_150.jsonl")
OUTPUT_PATH = Path("Outputs/parsed_prompts.jsonl")
VOCAB_DIR   = Path("vocab")

# Attribute match priority, first match specifies the attribute type.
ATTR_PRIORITY = ["shape", "material", "color"]

# Positional/directional words that are not real objects
STOP_OBJECTS = {
    "side", "left", "right", "top", "bottom", "front", "back",
    "center", "centre", "middle", "edge", "corner", "next", "out", "away"
}

# yields parsed records from .jsonl file
def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# saves list of dicts to a .jsonl file
def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

# loads a vocab .txt file and return single word set and multi word list
def load_vocab(path: Path):
    single, multi = set(), []
    if not path.exists():
        return single, multi
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = line.split("#")[0].strip().lower()
            if not entry:
                continue
            if " " in entry:
                multi.append(entry)
            else:
                single.add(entry)
    multi.sort(key=lambda x: -len(x.split()))
    return single, multi

# 
def load_all_vocabs(vocab_dir: Path):
    """Loads all vocab files — spatial.txt goes to spatial vocab, rest become attribute types ordered by ATTR_PRIORITY."""
    raw = {}
    spatial = (set(), [])

    vocab_dir = Path(vocab_dir)
    if not vocab_dir.exists():
        return [], spatial

    for fpath in sorted(vocab_dir.glob("*.txt")):
        stem = fpath.stem.lower()
        single, multi = load_vocab(fpath)
        if stem == "spatial":
            spatial = (single, multi)
        else:
            type_name = stem[:-1] if stem.endswith("s") and len(stem) > 1 else stem
            raw[type_name] = (single, multi)

    # Order by priority: shape > material > color > (alphabetical rest)
    ordered = []
    for ptype in ATTR_PRIORITY:
        if ptype in raw:
            s, m = raw.pop(ptype)
            ordered.append((ptype, s, m))
    for name in sorted(raw):
        s, m = raw[name]
        ordered.append((name, s, m))

    return ordered, spatial

# ─── SpaCy dep-tree helpers ──────────────────────────────────────────

def _resolve_subject_for_verb(verb_tok):
    """Walks up conjoined verbs to find the real noun subject."""
    for ch in verb_tok.children:
        if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"):
            return ch
    if verb_tok.dep_ == "conj":
        return _resolve_subject_for_verb(verb_tok.head)
    return None

def _pobj_of_prep(head_tok):
    """Finds the real noun after a positional word — e.g. 'left of table' returns 'table'."""
    # Direct child: left -> prep "of" -> pobj "table"
    for ch in head_tok.children:
        if ch.dep_ == "prep" and ch.lemma_.lower() in ("of", "to"):
            for ch2 in ch.children:
                if ch2.dep_ == "pobj" and ch2.pos_ in ("NOUN", "PROPN", "PRON", "VERB"):
                    return ch2
    # Sibling prep at the same level
    if head_tok.head is not None:
        for sib in head_tok.head.children:
            if sib.dep_ == "prep" and sib.lemma_.lower() in ("of", "to") and sib.i > head_tok.i:
                for ch2 in sib.children:
                    if ch2.dep_ == "pobj" and ch2.pos_ in ("NOUN", "PROPN", "PRON", "VERB"):
                        return ch2
    # Forward scan up to 5 tokens
    doc = head_tok.doc
    for i in range(head_tok.i + 1, min(head_tok.i + 5, len(doc))):
        tok = doc[i]
        if tok.pos_ == "ADP" and tok.lemma_.lower() in ("of", "to"):
            for ch in tok.children:
                if ch.dep_ == "pobj" and ch.pos_ in ("NOUN", "PROPN", "PRON", "VERB"):
                    return ch
        if tok.pos_ in ("NOUN", "PROPN") and tok.dep_ != "compound":
            break
    return None

def _best_modifier(tok):
    """Returns a directional modifier (e.g. 'left', 'right') from a token's children if present."""
    for ch in tok.children:
        if ch.dep_ == "amod" and ch.pos_ == "ADJ":
            if ch.lemma_.lower() in STOP_OBJECTS:
                return ch.lemma_.lower()
    for ch in tok.children:
        if ch.dep_ == "compound" and ch.pos_ in ("NOUN", "PROPN"):
            if ch.lemma_.lower() in STOP_OBJECTS:
                return ch.lemma_.lower()
    return None

def _anchor_noun(tok):
    """Resolves a token to its nearest relevant noun — handles verbs, aux, and head fallback."""
    if tok is None:
        return None
    if tok.pos_ in ("NOUN", "PROPN"):
        return tok
    if tok.pos_ in ("VERB", "AUX"):
        for ch in tok.children:
            if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"):
                return ch
        for ch in tok.children:
            if ch.dep_ in ("dobj", "obj") and ch.pos_ in ("NOUN", "PROPN"):
                return ch
        if tok.dep_ in ("acl", "relcl") and tok.head.pos_ in ("NOUN", "PROPN"):
            return tok.head
    if tok.head is not None and tok.head != tok:
        if tok.head.pos_ in ("NOUN", "PROPN"):
            return tok.head
    return None

def _classify_prep(prep, spatial_vocab):
    """Always returns 'spatial' — non-spatial detection is disabled."""
    return "spatial"

# ─── Attribute classifier ────────────────────────────────────────────

def classify_attribute(lemma, left_toks, tok_index, attr_vocabs):
    """Matches a token against vocab files and returns (type_name, matched_value, tokens_consumed). Multi-word checked first, then single."""
    # Multi-word check (longest-first, priority-ordered)
    for type_name, single_set, multi_list in attr_vocabs:
        for mw_entry in multi_list:
            mw_words = mw_entry.split()
            mw_len = len(mw_words)
            if tok_index + mw_len <= len(left_toks):
                candidate = " ".join(
                    left_toks[tok_index + j].lemma_.lower()
                    for j in range(mw_len)
                )
                if candidate == mw_entry:
                    return type_name, mw_entry, mw_len

    # Single-word check
    for type_name, single_set, multi_list in attr_vocabs:
        if lemma in single_set:
            return type_name, lemma, 1

    return "other", lemma, 1

# ─── Core extraction ─────────────────────────────────────────────────

def extract_all(doc, attr_vocabs, spatial_vocab):
    """Main extraction — returns objects, attributes, and spatial relations from a SpaCy doc."""

    # ── 1. Objects: noun chunks whose root is a real noun, not a positional word ──
    objects = []
    name_to_id = {}
    seen = set()
    counter = 0

    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        name = root.lemma_.lower()
        if name in STOP_OBJECTS or name in seen:
            continue
        # Skip "yarn" in "ball of yarn" — Y in "X of Y" when X is already a known object
        if (root.dep_ == "pobj" and
                root.head.dep_ == "prep" and root.head.lemma_ == "of" and
                root.head.head.lemma_.lower() in seen):
            continue
        seen.add(name)
        counter += 1
        oid = f"o{counter}"
        name_to_id[name] = oid
        objects.append({"id": oid, "name": name, "surface": chunk.text.lower()})

    # Supplementary pass: catch SpaCy-misclassified nouns tagged as PRON or VERB in pobj position
    for token in doc:
        if token.dep_ != "pobj" or token.pos_ in ("NOUN", "PROPN"):
            continue
        if token.pos_ not in ("PRON", "VERB"):
            continue
        name = token.lemma_.lower()
        if name in STOP_OBJECTS or name in seen:
            continue
        seen.add(name)
        counter += 1
        oid = f"o{counter}"
        name_to_id[name] = oid
        objects.append({"id": oid, "name": name, "surface": token.text.lower()})

    # ── 2. Attributes: left-side modifiers of each object noun ──
    attributes = []
    seen_objs = set()

    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in ("NOUN", "PROPN"):
            continue
        obj_name = root.lemma_.lower()
        if obj_name in STOP_OBJECTS or obj_name in seen_objs or obj_name not in name_to_id:
            continue
        seen_objs.add(obj_name)
        oid = name_to_id[obj_name]

        left_toks = list(root.lefts)
        consumed = set()

        # Pass 1: multi-word attribute matches
        for i in range(len(left_toks)):
            if i in consumed:
                continue
            tok = left_toks[i]
            if tok.dep_ not in ("amod", "compound", "nummod"):
                continue
            if tok.pos_ not in ("ADJ", "NOUN", "PROPN", "NUM"):
                continue
            tname, val, n = classify_attribute(tok.lemma_.lower(), left_toks, i, attr_vocabs)
            if n > 1 and tname != "other":
                attributes.append({"obj": oid, "type": tname, "value": val})
                for j in range(n):
                    consumed.add(i + j)

        # Pass 2: single-token attribute matches
        for i, tok in enumerate(left_toks):
            if i in consumed:
                continue

            if tok.dep_ == "nummod":
                try:
                    val = int(tok.text)
                except ValueError:
                    val = tok.text.lower()
                attributes.append({"obj": oid, "type": "count", "value": val})

            elif tok.dep_ == "amod" and tok.pos_ in ("ADJ", "NOUN", "PROPN"):
                tname, val, _ = classify_attribute(tok.lemma_.lower(), left_toks, i, attr_vocabs)
                if tname != "other":
                    attributes.append({"obj": oid, "type": tname, "value": val})

            elif tok.dep_ == "compound" and tok.pos_ in ("NOUN", "PROPN"):
                tname, val, _ = classify_attribute(tok.lemma_.lower(), left_toks, i, attr_vocabs)
                if tname != "other":
                    attributes.append({"obj": oid, "type": tname, "value": val})

    # ── 3. Relations: prepositions and verbs with spatial prep children ──
    relations = []
    seen_rels = set()

    def _add_rel(subj_name, rel, obj_name, rel_type, source="prep"):
        """Adds a relation only if both subject and object are known objects and not duplicates."""
        if obj_name is None:
            return
        sid = name_to_id.get(subj_name)
        oid = name_to_id.get(obj_name)
        if sid is None or oid is None:
            return
        key = (sid, rel, oid)
        if key in seen_rels:
            return
        seen_rels.add(key)
        relations.append({
            "subject": sid, "rel": rel,
            "object": oid, "type": rel_type,
            "source": source
        })

    # Preposition loop — handles "on X", "near X", "on the right of X", "next to X"
    for token in doc:
        if token.pos_ != "ADP":
            continue
        rel = token.lemma_.lower()

        pobj = None
        for ch in token.children:
            if ch.dep_ == "pobj" and ch.pos_ in ("NOUN", "PROPN", "PRON", "VERB"):
                pobj = ch
                break
        if pobj is None:
            continue

        # "next to X": token.head = "next" (ADJ/ADV in STOP_OBJECTS) — go one level up to find subject
        head = token.head
        if head.lemma_.lower() in STOP_OBJECTS and head.pos_ in ("ADV", "ADJ"):
            upper_subj = _anchor_noun(head.head)
            if upper_subj and upper_subj.lemma_.lower() not in STOP_OBJECTS:
                obj = pobj.lemma_.lower()
                if obj not in STOP_OBJECTS:
                    comp_rel = f"{head.lemma_.lower()} {rel}"
                    _add_rel(upper_subj.lemma_.lower(), comp_rel, obj, _classify_prep(comp_rel, spatial_vocab))
            continue

        subj_tok = _anchor_noun(token.head)
        if subj_tok is None:
            continue

        subj = subj_tok.lemma_.lower()
        obj  = pobj.lemma_.lower()

        if subj in STOP_OBJECTS:
            continue

        DIRECTIONAL = {"left", "right", "side"}
        if obj in STOP_OBJECTS:
            # "on the right side of table" -> right(subj, table)
            ref_tok = _pobj_of_prep(pobj)
            if ref_tok:
                ref_obj = ref_tok.lemma_.lower()
                if ref_obj not in STOP_OBJECTS:
                    mod = _best_modifier(pobj)
                    if mod:
                        base_rel = mod
                    elif obj in DIRECTIONAL:
                        base_rel = obj
                    else:
                        base_rel = f"{rel} {obj}"
                    rtype = _classify_prep(base_rel, spatial_vocab)
                    _add_rel(subj, base_rel, ref_obj, rtype)
            continue

        # "next to X" compound relation
        head = token.head
        head_lem = head.lemma_.lower()
        if head_lem in STOP_OBJECTS and head.pos_ in ("ADV", "ADJ", "NOUN"):
            comp_rel = f"{head_lem} {rel}"
            rtype = _classify_prep(rel, spatial_vocab)
            _add_rel(subj, comp_rel, obj, rtype)
            continue

        rtype = _classify_prep(rel, spatial_vocab)
        _add_rel(subj, rel, obj, rtype)

    # Verb loop — handles "mounted on wall", "lying on grass", "parked next to truck"
    for v in doc:
        if v.pos_ != "VERB":
            continue

        subj = None
        for ch in v.children:
            if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"):
                subj = ch
                break
        if subj is None and v.dep_ in ("acl", "relcl", "amod"):
            if v.head.pos_ in ("NOUN", "PROPN"):
                subj = v.head
        if subj is None and v.dep_ == "conj":
            subj = _resolve_subject_for_verb(v)

        if subj is None:
            continue

        subj_name = subj.lemma_.lower()
        if subj_name in STOP_OBJECTS:
            continue

        for ch in v.children:
            if ch.dep_ == "prep" and ch.pos_ == "ADP":
                prep_lem = ch.lemma_.lower()
                for pobj in ch.children:
                    if pobj.dep_ == "pobj" and pobj.pos_ in ("NOUN", "PROPN"):
                        obj_name = pobj.lemma_.lower()
                        if obj_name in STOP_OBJECTS:
                            continue
                        rtype = _classify_prep(prep_lem, spatial_vocab)
                        _add_rel(subj_name, prep_lem, obj_name, rtype)

            # "parked next to truck" — verb -> advmod "next" -> prep "to" -> pobj
            if ch.dep_ in ("advmod", "acomp", "amod") and ch.lemma_.lower() in STOP_OBJECTS:
                mod_lem = ch.lemma_.lower()
                for prep_ch in ch.children:
                    if prep_ch.dep_ == "prep" and prep_ch.pos_ == "ADP":
                        prep_lem = prep_ch.lemma_.lower()
                        comp_rel = f"{mod_lem} {prep_lem}"
                        for pobj in prep_ch.children:
                            if pobj.dep_ == "pobj" and pobj.pos_ in ("NOUN", "PROPN"):
                                obj_name = pobj.lemma_.lower()
                                if obj_name not in STOP_OBJECTS:
                                    _add_rel(subj_name, comp_rel, obj_name, "spatial")

    return {"objects": objects, "attributes": attributes, "relations": relations}

# ─── Pretty printer ──────────────────────────────────────────────────

def print_node(parsed):
    """Prints parsed prompt result — objects, attributes, relations — in readable format."""
    pid    = parsed["id"]
    prompt = parsed["prompt"]

    print()
    print(f"  {pid}")
    print(f"  {prompt}")
    print(f"  {'─' * len(prompt)}")

    print()
    print("  OBJECTS")
    for o in parsed["objects"]:
        print(f"      {o['id']:4s}  {o['name']:15s}  \"{o['surface']}\"")

    attrs = parsed.get("attributes", [])
    if attrs:
        print()
        print("  ATTRIBUTES")
        for a in attrs:
            print(f"      {a['obj']:4s}  {a['type']:10s}  {a['value']}")

    rels = parsed.get("relations", [])
    if rels:
        print()
        print("  RELATIONS")
        for r in rels:
            print(f"      {r['subject']:4s}  {r['rel']:15s}  {r['object']:4s}  [{r['type']}]")

    print()

# ─── Entry point ─────────────────────────────────────────────────────

def main():
    """Runs the prompt parser from CLI — loads spaCy, parses all prompts, saves to output."""
    import argparse
    ap = argparse.ArgumentParser(description="T2I Prompt Parser")
    ap.add_argument("--input",     type=Path, default=INPUT_PATH)
    ap.add_argument("--output",    type=Path, default=OUTPUT_PATH)
    ap.add_argument("--vocab-dir", type=Path, default=VOCAB_DIR)
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_sm")
    attr_vocabs, spatial = load_all_vocabs(args.vocab_dir)

    if not attr_vocabs:
        print(f"WARNING: No vocab files found in {args.vocab_dir} — attributes will not be typed.")

    print(f"Extracting triples from {args.input} ...")

    results = []
    for rec in load_jsonl(args.input):
        doc = nlp(rec["prompt"])
        extracted = extract_all(doc, attr_vocabs, spatial)
        parsed = {"id": rec["id"], "prompt": rec["prompt"], **extracted}

        if not parsed["attributes"]:
            del parsed["attributes"]
        if not parsed["relations"]:
            del parsed["relations"]

        results.append(parsed)

    save_jsonl(results, args.output)
    print(f"Done — {len(results)} prompts parsed → {args.output}")

if __name__ == "__main__":
    main()