import json
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

COMFY_HOST       = "127.0.0.1"
COMFY_PORT       = 8188
COMFY            = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_OUTPUT_DIR = Path("ComfyUI/output")

JSONL_PATH = Path("data/promts_150.jsonl")          # prompt input file
WORKFLOW_PATH = Path("Models/Sdxl_imageGeneration.json")    #Model workflow built on comfuui
OUT_DIR = Path("Outputs/Images_generated")         # output

POSITIVE_NODE_ID = None   

OUT_DIR.mkdir(parents=True, exist_ok=True)

def http_json(method: str, url: str, payload=None, timeout=60):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body}") from None

def http_bytes(url: str, timeout=60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_clip_text_nodes(workflow: dict):
    """Finding all CLIPTextEncode nodes with inputs.text"""
    hits = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and "text" in inputs:
                hits.append(str(node_id))
    return hits


def find_positive_node(workflow: dict):
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "KSampler":
            pos = node.get("inputs", {}).get("positive")
            if isinstance(pos, list) and len(pos) >= 1:
                return str(pos[0])
    return None


def find_saveimage_nodes(workflow: dict):
    """Finding SaveImage nodes to set filename_prefix."""
    hits = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "SaveImage":
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and "filename_prefix" in inputs:
                hits.append(str(node_id))
    return hits


def queue_prompt(workflow_api: dict) -> str:
    """POST /prompt — returns the server-side prompt_id."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow_api, "client_id": client_id}
    r = http_json("POST", f"{COMFY}/prompt", payload)
    if not r or "prompt_id" not in r:
        raise RuntimeError(f"Unexpected /prompt response: {r}")
    return r["prompt_id"]


def wait_for_done(prompt_id: str, poll_s=1.0, timeout_s=600):
    t0 = time.time()
    while True:
        try:
            hist = http_json("GET", f"{COMFY}/history/{prompt_id}", None, timeout=30)
        except Exception:
            hist = None

        if isinstance(hist, dict):
            info = hist.get(prompt_id)
            if info:
                # Check for errors
                status = info.get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI error for {prompt_id}: {msgs}")
                # Check for outputs
                if info.get("outputs"):
                    return info

        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")
        time.sleep(poll_s)


def download_outputs(info: dict, out_base: str):
    """Download generated images from ComfyUI and save locally."""
    outputs = info.get("outputs", {})
    saved = []
    idx = 1

    for _node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        images = node_out.get("images", [])
        for im in images:
            filename = im.get("filename")
            subfolder = im.get("subfolder", "")
            ftype = im.get("type", "output")
            if not filename:
                continue

            qs = {"filename": filename, "subfolder": subfolder, "type": ftype}
            url = f"{COMFY}/view?{urllib.parse.urlencode(qs)}"
            data = http_bytes(url, timeout=120)

            ext = os.path.splitext(filename)[1] or ".png"
            out_path = OUT_DIR / f"{out_base}_{idx:04d}{ext}"
            out_path.write_bytes(data)
            saved.append(out_path)
            idx += 1
            src = COMFY_OUTPUT_DIR / subfolder / filename
            try:
                src.unlink()
            except OSError:
                pass

    return saved

def main():
    # Load workflow
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    # Normalize: some exports wrap the graph under "prompt"
    if isinstance(workflow, dict) and "prompt" in workflow and isinstance(workflow["prompt"], dict):
        prompt_graph = workflow["prompt"]
    else:
        prompt_graph = workflow

    # Identify nodes
    save_nodes = find_saveimage_nodes(prompt_graph)
    if not save_nodes:
        raise RuntimeError("No SaveImage node found. Make sure your workflow ends with SaveImage.")

    # Determine positive prompt node
    if POSITIVE_NODE_ID:
        pos_node = POSITIVE_NODE_ID
    else:
        # Try auto-detect from KSampler connection
        pos_node = find_positive_node(prompt_graph)
        if not pos_node:
            # Fallback: use all CLIPTextEncode nodes
            all_text = find_clip_text_nodes(prompt_graph)
            if not all_text:
                raise RuntimeError("No CLIPTextEncode nodes found in workflow.")
            print(f"Could not auto-detect positive node. Will set ALL text nodes: {all_text}")
            pos_node = None  # signal to set all
        else:
            print(f"Auto-detected positive prompt node: {pos_node}")

    all_text_nodes = find_clip_text_nodes(prompt_graph) if pos_node is None else None
    print(f" SaveImage nodes: {save_nodes}")
    print(f"---")

    # Process each prompt
    total = 0
    errors = 0

    for rec in load_jsonl(JSONL_PATH):
        pid = str(rec.get("id", "")).strip()
        prompt_text = str(rec.get("prompt", "")).strip()
        if not pid or not prompt_text:
            print(f"Skipping invalid record: {rec}")
            continue

        # Deep copy the workflow for this prompt
        g = json.loads(json.dumps(prompt_graph))

        # Set positive prompt text
        if pos_node:
            g[pos_node]["inputs"]["text"] = prompt_text
        else:
            for nid in all_text_nodes:
                g[nid]["inputs"]["text"] = prompt_text

        # Set filename prefix to the prompt id
        for nid in save_nodes:
            g[nid]["inputs"]["filename_prefix"] = pid

        # Queue and wait
        try:
            print(f"[{pid}] Queuing: {prompt_text[:80]}...")
            server_pid = queue_prompt(g)
            info = wait_for_done(server_pid, timeout_s=1200)
            saved = download_outputs(info, out_base=pid)
            print(f"[{pid}] Saved {len(saved)} image(s): {[str(p) for p in saved]}")
            total += 1
        except Exception as e:
            print(f"[{pid}] ✗ ERROR: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Total Images Generated: {total} and Errors: {errors}")


if __name__ == "__main__":
    main()