import os
import re
import json
import numpy as np
import argparse
from typing import List, Dict, Any, Optional, Tuple


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _is_heading(line: str) -> Optional[Dict[str, Any]]:
    """Return {'level': int, 'title': str} if line is a markdown heading (#,##,...), else None."""
    m = re.match(r"^\s*(#{1,6})\s+(.*)$", line)
    if m:
        return {"level": len(m.group(1)), "title": m.group(2).strip()}
    return None


def _extract_html_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract a full <table>...</table> block starting at start_idx. Returns (block, next_index)."""
    buf = []
    i = start_idx
    first_line = lines[i]
    buf.append(first_line)

    if "</table>" in first_line.lower():
        return "\n".join(buf), i + 1

    i += 1
    while i < len(lines):
        buf.append(lines[i])
        if "</table>" in lines[i].lower():
            i += 1
            break
        i += 1
    return "\n".join(buf), i


def _extract_md_table(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Extract a markdown pipe-style table. Consecutive lines starting with '|' or optional align lines."""
    buf = []
    i = start_idx
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*\|.*\|\s*$", line) or re.match(
            r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", line
        ):
            buf.append(line)
            i += 1
        else:
            break
    return "\n".join(buf), i


def _extract_image(line: str, md_dir: str) -> Optional[Dict[str, Any]]:
    """Detect <img ...> or ![]() in a line and return a paragraph dict with absolute image_path."""
    # HTML image tag
    m = re.search(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*", line, re.IGNORECASE)
    if m:
        src = m.group(1).strip()
        alt_m = re.search(r"alt=[\"']([^\"']+)[\"']", line, re.IGNORECASE)
        alt_text = alt_m.group(1).strip() if alt_m else ""
        if src.startswith(("http://", "https://", "data:")):
            image_path = src
        else:
            image_path = os.path.normpath(os.path.join(md_dir, src))
        return {"type": "image", "content": alt_text, "image_path": image_path}

    # Markdown image syntax ![alt](src)
    m2 = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", line)
    if m2:
        alt_text = m2.group(1).strip()
        src = m2.group(2).strip()
        if src.startswith(("http://", "https://", "data:")):
            image_path = src
        else:
            image_path = os.path.normpath(os.path.join(md_dir, src))
        return {"type": "image", "content": alt_text, "image_path": image_path}

    return None


def parse_markdown_to_corpus(md_path: str) -> Dict[str, Any]:
    """
    Parse a Markdown file into a corpus JSON compatible with MuReAct_TF DocIndex:
    - Recognizes headings (#, ##, ...). If the minimum heading level > 1, normalize so min level becomes 1.
    - Treats each table (HTML <table>...</table> or pipe-style markdown) as a single paragraph string.
    - Treats each image (<img ...> / ![]()) as a dict paragraph: {type: 'image', content: alt, image_path: absolute_path}.
    - Other text blocks are grouped by blank lines.
    - Maintains parent-child hierarchy with ids (flat sequential numbering: "1", "2", "3", ...).
    """
    md_dir = os.path.dirname(md_path)
    content = _read_file(md_path)
    lines = content.splitlines()

    # Pass 1: find minimum heading level present to normalize hierarchy
    heading_levels = []
    for ln in lines:
        info = _is_heading(ln)
        if info:
            heading_levels.append(info["level"])
    min_level = min(heading_levels) if heading_levels else 1
    level_offset = min_level - 1

    nodes: List[Dict[str, Any]] = []
    node_map: Dict[str, Dict[str, Any]] = {}
    stack: List[Dict[str, Any]] = []  # each: {'id': str, 'level': int}
    next_id = 0

    def _alloc_id() -> str:
        nonlocal next_id
        next_id += 1
        return str(next_id)

    front_matter_id: Optional[str] = None

    def _ensure_front_matter(paragraph: Any):
        nonlocal front_matter_id, nodes, node_map, stack
        if front_matter_id is None:
            front_matter_id = _alloc_id()
            fm = {"id": front_matter_id, "title": "前言", "paragraphs": [], "children": []}
            nodes.append(fm)
            node_map[front_matter_id] = fm
            stack = [{"id": front_matter_id, "level": 1}]
        node_map[front_matter_id]["paragraphs"].append(paragraph)

    def _new_node(level: int, title: str) -> Dict[str, Any]:
        nonlocal stack, nodes, node_map

        # Top-level node or empty stack
        if level == 1 or not stack:
            node_id = _alloc_id()
            node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        # Pop until we find parent at level-1
        while stack and stack[-1]["level"] >= level:
            stack.pop()

        if not stack:
            # No valid parent, treat as top-level
            node_id = _alloc_id()
            node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
            nodes.append(node)
            node_map[node_id] = node
            stack = [{"id": node_id, "level": 1}]
            return node

        parent = stack[-1]
        parent_id = parent["id"]

        node_id = _alloc_id()
        node = {"id": node_id, "title": title.strip(), "paragraphs": [], "children": []}
        nodes.append(node)
        node_map[node_id] = node

        # Link to immediate parent
        node_map[parent_id]["children"].append(node_id)
        stack.append({"id": node_id, "level": level})
        return node

    def _append_paragraph(node: Optional[Dict[str, Any]], paragraph: Any):
        if node is None:
            _ensure_front_matter(paragraph)
            return
        node["paragraphs"].append(paragraph)

    current_node: Optional[Dict[str, Any]] = None
    i = 0
    while i < len(lines):
        line = lines[i]
        heading = _is_heading(line)

        if heading:
            normalized = max(1, heading["level"] - level_offset)
            if stack:
                parent_level = stack[-1]["level"]
                if normalized > parent_level + 1:
                    level = parent_level + 1  # avoid multi-level jumps
                else:
                    level = normalized
            else:
                level = normalized

            current_node = _new_node(level, heading["title"])
            i += 1
            continue

        # HTML table block
        if "<table" in line.lower():
            block, nxt = _extract_html_table(lines, i)
            _append_paragraph(current_node, block)
            i = nxt
            continue

        # pipe-style markdown table block
        if re.match(r"^\s*\|.*\|\s*$", line):
            block, nxt = _extract_md_table(lines, i)
            _append_paragraph(current_node, block)
            i = nxt
            continue

        # image block
        img_para = _extract_image(line, md_dir)
        if img_para is not None:
            _append_paragraph(current_node, img_para)
            i += 1
            continue

        # collect text paragraph until blank line or block boundary
        if line.strip() == "":
            i += 1
            continue

        buf = [line]
        i += 1
        while i < len(lines):
            peek = lines[i]
            if peek.strip() == "":
                i += 1
                break
            if (
                _is_heading(peek)
                or "<table" in peek.lower()
                or re.match(r"^\s*\|.*\|\s*$", peek)
                or _extract_image(peek, md_dir) is not None
            ):
                break
            buf.append(peek)
            i += 1

        paragraph_text = "\n".join(buf)
        _append_paragraph(current_node, paragraph_text)

    return {"nodes": nodes}


def main():
    parser = argparse.ArgumentParser(
        description="Process a PDF using PaddleOCRVL, merge Markdown/JSON, then parse Markdown to corpus JSON (optionally build embeddings)."
    )

    # --- Required: PDF input ---
    parser.add_argument("--input", type=str, required=True, help="Path to the input PDF file.")

    # --- Output directory prefix (default behavior unchanged) ---
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file prefix (without extension). Default: input path without extension.",
    )

    # --- PaddleOCRVL params (now configurable; defaults match your current hardcoded settings) ---
    parser.add_argument(
        "--paddle-vl-rec-backend",
        type=str,
        default="vllm-server",
        help="PaddleOCRVL vl_rec_backend. Default: vllm-server",
    )
    parser.add_argument(
        "--paddle-vl-rec-server-url",
        type=str,
        default="http://127.0.0.1:8956/v1",
        help="PaddleOCRVL vl_rec_server_url. Default: http://127.0.0.1:8956/v1",
    )

    # --- Embedding options (API key / base url / model path as requested) ---
    parser.add_argument("--build-embeddings", action="store_true")

    # Keep default identical to your current script
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "/home/lizhanli/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"),
        help="Embedding model name/path. Default: env EMBEDDING_MODEL or the existing local path.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        help="Batch size for embedding requests. Default: env EMBEDDING_BATCH_SIZE or 64.",
    )

    parser.add_argument(
        "--embed-api-key",
        type=str,
        default=os.getenv("EMBED_API_KEY", ""),
        help="Embedding API key (Bearer). Default: env EMBED_API_KEY (can be empty for local servers).",
    )
    parser.add_argument(
        "--embed-base-url",
        type=str,
        default=os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8756/v1"),
        help="Embedding API base URL. Default: env EMBED_BASE_URL or http://127.0.0.1:8756/v1",
    )

    args = parser.parse_args()

    # Enforce: input must be PDF
    input_lower = args.input.lower()
    if not input_lower.endswith(".pdf"):
        raise ValueError("Unsupported input format. This script now accepts PDF only (use .pdf).")

    # Output base path logic (unchanged)
    if args.output is None:
        output_base_path, _ = os.path.splitext(args.input)
    else:
        output_base_path = args.output

    # Create output directory
    os.makedirs(f"{output_base_path}/", exist_ok=True)

    print(f"Starting pipeline for PDF: {args.input}")

    # --- OCR PDF -> per-page Markdown/JSON -> merged ---
    print(f"Starting OCR pipeline for PDF: {args.input}")
    from paddleocr import PaddleOCRVL  # type: ignore

    pipeline = PaddleOCRVL(
        vl_rec_backend=args.paddle_vl_rec_backend,
        vl_rec_server_url=args.paddle_vl_rec_server_url,
    )
    output = pipeline.predict(args.input)

    basename = output_base_path.split("/")[-1]
    merged_json_path = f"{output_base_path}/{basename}.json"
    merged_md_path = f"{output_base_path}/{basename}.md"
    temp_json_path = f"{output_base_path}/{basename}_temp_page.json"
    temp_md_path = f"{output_base_path}/{basename}_temp_page.md"

    all_json_data = []
    print(f"\n--- Processing {len(output)} results and merging ---")

    # ensure merged md exists/cleared
    with open(merged_md_path, "w", encoding="utf-8"):
        pass

    for i, res in enumerate(output):
        print(f"  Processing page/result {i + 1}...")

        # JSON
        try:
            res.save_to_json(save_path=temp_json_path)
            with open(temp_json_path, "r", encoding="utf-8") as f_temp_json:
                page_data = json.load(f_temp_json)
                all_json_data.append(page_data)
        except Exception as e:
            print(f"    ! Error processing JSON for page {i + 1}: {e}")

        # Markdown
        try:
            res.save_to_markdown(save_path=temp_md_path)
            with open(temp_md_path, "r", encoding="utf-8") as f_temp_md:
                page_content = f_temp_md.read()
            with open(merged_md_path, "a", encoding="utf-8") as f_final_md:
                f_final_md.write(page_content)
                if i < len(output) - 1:
                    f_final_md.write("\n\n")
        except Exception as e:
            print(f"    ! Error processing Markdown for page {i + 1}: {e}")

    print(f"\n--- Saving merged JSON to {merged_json_path} ---")
    try:
        with open(merged_json_path, "w", encoding="utf-8") as f_final_json:
            json.dump(all_json_data, f_final_json, indent=2, ensure_ascii=False)
        print("  Successfully saved merged JSON.")
    except Exception as e:
        print(f"  ! Error saving merged JSON: {e}")

    # Cleanup temp files
    try:
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        if os.path.exists(temp_md_path):
            os.remove(temp_md_path)
        print("--- Temporary files cleaned up ---")
    except Exception as e:
        print(f"  ! Warning: could not remove temporary files: {e}")

    # --- Parse merged Markdown -> corpus ---
    print(f"\n--- Parsing Markdown to corpus: {merged_md_path} ---")
    corpus = parse_markdown_to_corpus(merged_md_path)

    # --- Optional: build embeddings (embed raw text directly) ---
    if args.build_embeddings:
        texts: List[str] = []
        id_map: List[Dict[str, Any]] = []

        for n in corpus.get("nodes", []):
            nid = n.get("id")
            pars = n.get("paragraphs", [])
            for pi, p in enumerate(pars):
                if isinstance(p, str):
                    t = p.strip()
                elif isinstance(p, dict):
                    t = str(p.get("content", "")).strip()
                else:
                    t = str(p).strip()

                if not t:
                    continue
                texts.append(t)
                id_map.append({"node_id": nid, "paragraph_index": pi})

        if texts:
            bs = max(1, int(args.embedding_batch_size))
            emb_list: List[List[float]] = []

            embed_api_key = args.embed_api_key
            embed_base_url = args.embed_base_url.rstrip("/")

            def _http_embed(model: str, inputs: List[str]) -> List[List[float]]:
                import requests

                url = embed_base_url + "/embeddings"
                headers = {"Content-Type": "application/json"}
                if embed_api_key:
                    headers["Authorization"] = f"Bearer {embed_api_key}"
                payload = {"model": model, "input": inputs}
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return [item.get("embedding") for item in data.get("data", [])]

            for j in range(0, len(texts), bs):
                batch = texts[j : j + bs]
                outs = _http_embed(args.embedding_model, batch)  # raw text
                emb_list.extend(outs)

            arr = np.asarray(emb_list, dtype=np.float32)
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norm + 1e-12)

            emb_path = f"{output_base_path}/{basename}_emb.npy"
            idmap_path = f"{output_base_path}/{basename}_idmap.json"

            np.save(emb_path, arr.astype(np.float16))
            with open(idmap_path, "w", encoding="utf-8") as f_id:
                json.dump(id_map, f_id, ensure_ascii=False)

            corpus["vector_store"] = {
                "matrix_path": emb_path,
                "id_map_path": idmap_path,
                "model_name": args.embedding_model,
                "normalized": True,
                "dtype": "float16",
                "embed_base_url": args.embed_base_url,
            }

    corpus_path = f"{output_base_path}/{basename}_corpus.json"
    try:
        with open(corpus_path, "w", encoding="utf-8") as f_c:
            json.dump(corpus, f_c, indent=2, ensure_ascii=False)
        print(f"--- Corpus JSON saved to {corpus_path} ---")
    except Exception as e:
        print(f"  ! Error saving corpus JSON: {e}")


if __name__ == "__main__":
    main()
