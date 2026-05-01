from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from .utils import _round_score


def http_embeddings(
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
    inputs: List[str],
    timeout: int = 120,
) -> List[List[float]]:
    if not api_key:
        raise RuntimeError("Please set EMBED_API_KEY (or pass embed_api_key)")
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": inputs}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    arr: List[List[float]] = []
    for item in data.get("data", []):
        emb = item.get("embedding")
        if isinstance(emb, list):
            arr.append(emb)
    return arr


def vector_search(
    doc_index: Any,
    query: str,
    scope: str = "full",
    doc_id: Optional[str] = None,
    top_k: int = 2,
    include_images: bool = True,
    embed_api_key: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    embed_model: Optional[str] = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    if not query:
        return {"ok": False, "error": "empty query"}
    if doc_index._vec_matrix is None or not len(doc_index._vec_idmap):
        return {"ok": False, "error": "vector_store not available"}

    model_name = embed_model or os.getenv("EMBEDDING_MODEL", doc_index._vec_model_name or "Qwen/Qwen3-Embedding-8B")
    api_key = embed_api_key or os.getenv("EMBED_API_KEY")
    base_url = (embed_base_url or os.getenv("EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    q_list = http_embeddings(api_key=api_key, base_url=base_url, model=model_name, inputs=[query])
    if not q_list:
        return {"ok": False, "error": "embedding_failed"}

    q_vec = np.asarray(q_list[0], dtype=np.float32)
    q_norm = float(np.linalg.norm(q_vec)) + 1e-12

    idxs = list(range(len(doc_index._vec_idmap)))
    if scope == "doc" and doc_id is not None:
        did = str(doc_id)
        idxs = [
            i
            for i, meta in enumerate(doc_index._vec_idmap)
            if str(meta.get("doc_id", "")) == did
        ]
        if not idxs:
            return {
                "ok": False,
                "error": f"no embeddings under doc '{did}'",
                "query": query,
                "scope": scope,
                "doc_id": did,
                "results": [],
            }

    matrix = doc_index._vec_matrix
    sims: List[float] = []
    for i in idxs:
        v = np.asarray(matrix[i], dtype=np.float32)
        v_norm = 1.0 if doc_index._vec_normalized else (float(np.linalg.norm(v)) + 1e-12)
        sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
        sims.append(max(-1.0, min(1.0, sim)))

    order = np.argsort(sims)[::-1]
    k = max(1, int(top_k))

    hits: List[Dict[str, Any]] = []
    for rank in order[:k]:
        global_idx = idxs[int(rank)]
        meta = doc_index._vec_idmap[global_idx]
        nid = str(meta.get("node_id"))
        did = str(meta.get("doc_id", ""))
        paragraph_index = int(meta.get("paragraph_index", 0))

        node = (doc_index.nodes_by_doc.get(did) or {}).get(nid) or {"paragraphs": []}
        text = ""
        paragraphs = node.get("paragraphs", [])
        if 0 <= paragraph_index < len(paragraphs):
            paragraph = paragraphs[paragraph_index]
            if isinstance(paragraph, str):
                text = paragraph
            elif isinstance(paragraph, dict):
                text = paragraph.get("content", "")
            else:
                text = str(paragraph)

        neighbors = doc_index._neighbor_context_for(
            did,
            nid,
            paragraph_index,
            include_images=include_images,
            neighbor_window=neighbor_window,
        )
        index_set = {paragraph_index}
        for item in neighbors:
            try:
                index_set.add(int(item["paragraph_index"]))
            except Exception:
                continue
        paragraph_indexes = sorted(index_set)

        hits.append(
            {
                "score": _round_score(sims[int(rank)]),
                "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": paragraph_indexes},
                "text": text,
                "neighbors": neighbors,
            }
        )

    return {
        "ok": True,
        "query": query,
        "scope": scope,
        "doc_id": str(doc_id) if doc_id is not None else None,
        "results": hits,
    }
