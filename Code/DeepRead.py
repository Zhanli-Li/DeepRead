#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import base64
import json
import math
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import tiktoken  # type: ignore

    _tiktoken_available = True
except Exception:  # pragma: no cover
    tiktoken = None
    _tiktoken_available = False


# ------------------------------
# Neighbor window helpers
# ------------------------------
def _normalize_neighbor_window(window: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if window is None:
        return None
    if not isinstance(window, tuple) or len(window) != 2:
        raise ValueError("neighbor_window must be a tuple of 2 integers, e.g. (1, -1) or (0, 0)")
    up, down = window
    if not isinstance(up, int) or not isinstance(down, int):
        raise ValueError("neighbor_window elements must be integers")
    if up < 0:
        raise ValueError(f"neighbor_window[0] (up) must be >= 0; got {up!r}")
    if down > 0:
        raise ValueError(f"neighbor_window[1] (down) must be <= 0; got {down!r}")
    if up == 0 and down == 0:
        return None
    return (up, down)


def _round_score(val: float) -> float:
    try:
        return float(f"{float(val):.2f}")
    except Exception:
        return float(val)


# ------------------------------
# Tool-call fallback parsing
# ------------------------------
_DSML_MARK_RE = re.compile(r"[\uFF5C|]DSML[\uFF5C|]")

_ALLOWED_TOOL_NAMES = {
    "read_section",
    "bm25_search",
    "regex_search",
    "vector_search",
    "hybrid_search",
    "semantic_retrieval",
}

_INLINE_TOOL_HEAD_RE = re.compile(
    r"\b(?P<name>"
    + "|".join(re.escape(n) for n in sorted(_ALLOWED_TOOL_NAMES, key=len, reverse=True))
    + r")\b\s*(?P<lp>\()?\s*[:=]?\s*\{",
    re.I,
)

_XML_INVOKE_START_RE = re.compile(
    r'<\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)\s+name\s*=\s*"([^"]+)"[^>]*>',
    re.I,
)

_XML_ANY_END_RE = re.compile(
    r"<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_calls|tool_calls|function_call|tool_call)[^>]*>",
    re.I,
)

_XML_PARAM_RE = re.compile(
    r'<\s*(?:parameter|param)\s+name\s*=\s*"([^"]+)"[^>]*>\s*'
    r"([\s\S]*?)"
    r"(?="
    r"<\s*/\s*(?:parameter|param)[^>]*>"
    r"|<\s*(?:parameter|param)\s+name\s*="
    r"|<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_calls|tool_calls|function_call|tool_call)[^>]*>"
    r"|$)",
    re.I,
)


def normalize_toolcall_markup(text: str) -> str:
    if not text:
        return text
    text = _DSML_MARK_RE.sub("", text)
    text = text.replace("\u200b", "")
    return text


def normalize_toolcall_markup_preserve_len(text: str) -> str:
    if not text:
        return text

    def _pad(m: re.Match) -> str:
        return " " * len(m.group(0))

    text = _DSML_MARK_RE.sub(_pad, text)
    text = text.replace("\u200b", " ")
    return text


def _extract_balanced_braces(s: str, brace_start: int) -> Optional[int]:
    if brace_start < 0 or brace_start >= len(s) or s[brace_start] != "{":
        return None

    depth = 0
    in_str = False
    esc = False
    quote_char = ""

    for i in range(brace_start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == quote_char:
                in_str = False
                quote_char = ""
            continue

        if ch == '"' or ch == "'":
            in_str = True
            quote_char = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _try_parse_json_obj(obj_text: str) -> Optional[Dict[str, Any]]:
    if not obj_text:
        return None
    t = obj_text.strip()
    t2 = t.replace('\\"', '"').replace("\\'", "'")

    try:
        v = json.loads(t2)
        return v if isinstance(v, dict) else None
    except Exception:
        pass

    try:
        v = ast.literal_eval(t2)
        return dict(v) if isinstance(v, dict) else None
    except Exception:
        return None


def fallback_tool_calls_from_text_xmlish(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    t = normalize_toolcall_markup(text)
    if ("<invoke" not in t.lower()) and ("<functioninvoke" not in t.lower()):
        return None
    if "<parameter" not in t.lower() and "<param" not in t.lower():
        return None

    calls: List[Dict[str, Any]] = []

    for m in _XML_INVOKE_START_RE.finditer(t):
        raw_name = (m.group(1) or "").strip()
        tool_name = raw_name.replace(" ", "_")
        if tool_name not in _ALLOWED_TOOL_NAMES:
            continue

        body_start = m.end()
        end_pos = len(t)
        end_m = _XML_ANY_END_RE.search(t, pos=body_start)
        if end_m:
            end_pos = end_m.start()

        body = t[body_start:end_pos]

        args: Dict[str, Any] = {}
        for pk, pv in _XML_PARAM_RE.findall(body):
            k = (pk or "").strip()
            v = (pv or "").strip()
            if k:
                args[k] = v

        calls.append(
            {
                "id": f"fallback_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )

    return calls or None


def fallback_tool_calls_from_text_inline_json(text: str) -> Optional[Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]]:
    if not text:
        return None

    t = normalize_toolcall_markup_preserve_len(text)

    calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []

    for m in _INLINE_TOOL_HEAD_RE.finditer(t):
        name = (m.group("name") or "").strip()
        if name not in _ALLOWED_TOOL_NAMES:
            continue

        brace_start = t.find("{", m.start())
        if brace_start < 0:
            continue
        brace_end = _extract_balanced_braces(t, brace_start)
        if brace_end is None:
            continue

        obj_text = t[brace_start : brace_end + 1]
        args = _try_parse_json_obj(obj_text)
        if not isinstance(args, dict):
            continue

        end_excl = brace_end + 1
        j = end_excl
        while j < len(t) and t[j].isspace():
            j += 1
        if (m.group("lp") is not None) and j < len(t) and t[j] == ")":
            end_excl = j + 1

        calls.append(
            {
                "id": f"fallback_{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
        spans.append((m.start(), end_excl))

    if not calls:
        return None
    return calls, spans


def fallback_tool_calls_from_text(text: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    if not text:
        return None

    calls_xml = fallback_tool_calls_from_text_xmlish(text)
    if calls_xml:
        return calls_xml, {"kind": "xmlish"}

    inline = fallback_tool_calls_from_text_inline_json(text)
    if inline:
        calls, spans = inline
        return calls, {"kind": "inline_json", "spans": spans}

    return None


def strip_function_calls_block_any(text: str) -> str:
    if not text:
        return text
    t = normalize_toolcall_markup(text)

    t = re.sub(
        r"<\s*(?:function_calls|tool_calls)[^>]*>[\s\S]*?<\s*/\s*(?:function_calls|tool_calls)[^>]*>",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"<\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)[^>]*>[\s\S]*?(?:<\s*/\s*(?:invoke|functioninvoke|toolinvoke|function_call|tool_call)[^>]*>|$)",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"<\s*/\s*(?:function_calls|tool_calls|invoke|functioninvoke|toolinvoke|parameter|param)[^>]*>",
        "",
        t,
        flags=re.I,
    )

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def strip_inline_tool_calls(text: str, spans: List[Tuple[int, int]]) -> str:
    if not text or not spans:
        return text
    spans_sorted = sorted(spans, key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    for s, e in spans_sorted:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    out_parts: List[str] = []
    last = 0
    for s, e in merged:
        out_parts.append(text[last:s])
        last = e
    out_parts.append(text[last:])

    cleaned = "".join(out_parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def should_sanitize_for_vllm(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    u = base_url.lower()
    if "openrouter.ai" in u:
        return False
    if "api.openai.com" in u:
        return False
    if "localhost" in u or "127.0.0.1" in u or "vllm" in u:
        return True
    return False


# ------------------------------
# Logging helper (JSONL)
# ------------------------------
class JsonlLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        if not os.path.exists(path):
            with open(self.path, "w", encoding="utf-8"):
                pass

    def log(self, event: str, **payload: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------
# Tokenization + counting
# ------------------------------
_ws_re = re.compile(r"\s+")


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return [t for t in _ws_re.split(text.strip()) if t]


def count_model_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        if tiktoken is None:
            raise ImportError
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(simple_tokenize(text))


# ------------------------------
# Corpus + retrieval
# ------------------------------
class DocIndex:
    def __init__(self, nodes: List[Dict[str, Any]], neighbor_window: Optional[Tuple[int, int]]) -> None:
        self.neighbor_window: Optional[Tuple[int, int]] = _normalize_neighbor_window(neighbor_window)

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.nodes_by_doc: Dict[str, Dict[str, Any]] = {}

        for n in nodes:
            nid = str(n["id"])
            doc_id = str(n.get("doc_id", ""))
            n.setdefault("title", nid)
            n.setdefault("paragraphs", [])
            n.setdefault("children", [])

            text_parts: List[str] = []
            for p in n["paragraphs"]:
                if isinstance(p, str):
                    text_parts.append(p)
                elif isinstance(p, dict):
                    content = p.get("content", "")
                    if content:
                        text_parts.append(content)
                else:
                    text_parts.append(str(p))
            full_text = "\n".join(text_parts) if text_parts else ""
            n["_tokens"] = simple_tokenize(full_text)
            n["_model_token_count"] = count_model_tokens(full_text)

            self.nodes[nid] = n
            self.nodes_by_doc.setdefault(doc_id, {})
            self.nodes_by_doc[doc_id][nid] = n

        self.node_to_doc_id: Dict[str, str] = {}
        for did, doc_nodes in self.nodes_by_doc.items():
            for nid in doc_nodes.keys():
                self.node_to_doc_id[str(nid)] = str(did)

        self.par_docs: List[Dict[str, Any]] = []
        for did, doc_nodes in self.nodes_by_doc.items():
            for nid, node in doc_nodes.items():
                for i, p in enumerate(node["paragraphs"]):
                    if isinstance(p, str):
                        text = p
                    elif isinstance(p, dict):
                        text = p.get("content", "")
                    else:
                        text = str(p)
                    if text:
                        toks = simple_tokenize(text)
                        self.par_docs.append(
                            {
                                "doc_id": str(did),
                                "node_id": str(nid),
                                "p_idx": i,
                                "text": text,
                                "tokens": toks,
                                "len": len(toks),
                            }
                        )

        self.N = len(self.par_docs) if self.par_docs else 1
        self.avgdl = sum(d["len"] for d in self.par_docs) / self.N

        df: Dict[str, int] = {}
        for d in self.par_docs:
            for term in set(d["tokens"]):
                df[term] = df.get(term, 0) + 1
        self.df = df

        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

        self._vec_matrix: Optional[np.ndarray] = None
        self._vec_idmap: List[Dict[str, Any]] = []
        self._vec_model_name: Optional[str] = None
        self._vec_normalized: bool = False

    def overview(self) -> str:
        lines: List[str] = []
        for nid, node in self.nodes.items():
            lines.append(
                f"- (doc_id={str(node.get('doc_id', ''))}) "
                f"[{nid}] {node['title']} | paragraphs={len(node['paragraphs'])} | "
                f"tokens={node.get('_model_token_count', len(node['_tokens']))} | children={node['children']}"
            )
        return "\n".join(lines)

    def _neighbor_context_for(
        self,
        doc_id: str,
        node_id: str,
        paragraph_index: int,
        include_images: bool = True,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        window = _normalize_neighbor_window(neighbor_window if neighbor_window is not None else self.neighbor_window)
        node = (self.nodes_by_doc.get(doc_id) or {}).get(node_id) or {"paragraphs": []}
        npars = len(node.get("paragraphs", []))

        if window is None or npars == 0:
            return []

        up, down = window
        up = max(0, up)
        down = abs(down)

        neighbors: List[Dict[str, Any]] = []

        if up > 0 and paragraph_index > 0:
            start = max(0, paragraph_index - up)
            end = paragraph_index
            if start < end:
                prev_section = self.read_section(
                    doc_id=doc_id,
                    node_id=node_id,
                    start_paragraph=start,
                    end_paragraph=end,
                    include_images=include_images,
                )
                if prev_section.get("ok", True):
                    neighbors.extend(prev_section.get("paragraphs", []))

        if down > 0 and paragraph_index + 1 < npars:
            start = paragraph_index + 1
            end = min(npars, paragraph_index + 1 + down)
            if start < end:
                next_section = self.read_section(
                    doc_id=doc_id,
                    node_id=node_id,
                    start_paragraph=start,
                    end_paragraph=end,
                    include_images=include_images,
                )
                if next_section.get("ok", True):
                    neighbors.extend(next_section.get("paragraphs", []))

        return neighbors

    def read_section(
        self,
        doc_id: Optional[str],
        node_id: str,
        start_paragraph: int,
        end_paragraph: int,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        doc_id_str = str(doc_id or "")
        node_id_str = str(node_id) if node_id is not None else ""

        node = (self.nodes_by_doc.get(doc_id_str) or {}).get(node_id_str)
        if not node:
            return {
                "ok": False,
                "error": f"node_id '{node_id_str}' not found in doc '{doc_id_str}'",
                "ref": {"doc_id": doc_id_str, "node_id": node_id_str, "paragraph_indexes": []},
                "paragraphs": [],
            }

        paragraphs = node.get("paragraphs", [])
        n = len(paragraphs)

        start_paragraph = max(0, min(start_paragraph, n))
        if end_paragraph == -1:
            end_paragraph = n
        end_paragraph = max(start_paragraph, min(end_paragraph, n))

        slice_paragraphs = paragraphs[start_paragraph:end_paragraph]

        out_items: List[Dict[str, Any]] = []
        for rel_idx, p in enumerate(slice_paragraphs):
            g_idx = start_paragraph + rel_idx

            if isinstance(p, str):
                if p.strip():
                    out_items.append({"paragraph_index": g_idx, "type": "text", "text": p})
                continue

            if isinstance(p, dict):
                p_type = p.get("type", "text")
                content = p.get("content", "")

                if p_type == "image" and include_images:
                    if content:
                        out_items.append({"paragraph_index": g_idx, "type": "text", "text": f"[image_ocr]{content}[/image_ocr]"})
                    image_path = p.get("image_path")
                    if image_path and Path(image_path).exists():
                        try:
                            with open(image_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            mime_type, _ = mimetypes.guess_type(image_path)
                            if mime_type:
                                out_items.append(
                                    {
                                        "paragraph_index": g_idx,
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                                    }
                                )
                        except Exception:
                            pass
                    else:
                        if content:
                            out_items.append({"paragraph_index": g_idx, "type": "text", "text": f"[image_ocr]{content}[/image_ocr]"})
                else:
                    if content:
                        out_items.append({"paragraph_index": g_idx, "type": "text", "text": content})
            else:
                out_items.append({"paragraph_index": g_idx, "type": "text", "text": str(p)})

        para_indexes = sorted({item["paragraph_index"] for item in out_items})
        return {"ok": True, "ref": {"doc_id": doc_id_str, "node_id": node_id_str, "paragraph_indexes": para_indexes}, "paragraphs": out_items}

    def bm25_search(
        self,
        query: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        top_k: int = 2,
        k1: float = 1.5,
        b: float = 0.75,
        include_images: bool = True,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        if not query:
            return {"ok": False, "error": "empty query"}

        q_terms = simple_tokenize(query)
        if not q_terms:
            return {"ok": False, "error": "empty query"}

        docs = self.par_docs
        if scope == "doc" and doc_id is not None:
            did = str(doc_id)
            docs = [d for d in self.par_docs if d["doc_id"] == did]
            if not docs:
                return {
                    "ok": False,
                    "error": f"no paragraphs under doc '{did}'",
                    "query": query,
                    "scope": scope,
                    "doc_id": did,
                    "results": [],
                }

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for d in docs:
            score_val = 0.0
            dl = d["len"]
            tf_count: Dict[str, int] = {}
            for t in d["tokens"]:
                tf_count[t] = tf_count.get(t, 0) + 1
            for term in q_terms:
                if term not in tf_count:
                    continue
                idf_val = self.idf.get(term, 0.0)
                tf = tf_count[term]
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                score_val += idf_val * (tf * (k1 + 1)) / (denom + 1e-9)
            if score_val > 0:
                scores.append((score_val, d))

        scores.sort(key=lambda x: x[0], reverse=True)

        hits: List[Dict[str, Any]] = []
        k = max(1, int(top_k))

        for raw_score, d in scores[:k]:
            did = d["doc_id"]
            nid = d["node_id"]
            p_idx = d["p_idx"]

            neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
            index_set = {p_idx}
            for item in neighbors:
                try:
                    index_set.add(int(item["paragraph_index"]))
                except Exception:
                    continue
            para_indexes = sorted(index_set)

            hits.append(
                {
                    "score": _round_score(raw_score),
                    "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                    "text": d["text"],
                    "neighbors": neighbors,
                }
            )

        return {"ok": True, "query": query, "scope": scope, "doc_id": str(doc_id) if doc_id is not None else None, "results": hits}

    def _http_embeddings(
        self,
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
        self,
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
        if self._vec_matrix is None or not len(self._vec_idmap):
            return {"ok": False, "error": "vector_store not available"}

        model_name = embed_model or os.getenv("EMBEDDING_MODEL", self._vec_model_name or "Qwen/Qwen3-Embedding-8B")
        api_key = embed_api_key or os.getenv("EMBED_API_KEY")
        base_url = (embed_base_url or os.getenv("EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

        q_list = self._http_embeddings(api_key=api_key, base_url=base_url, model=model_name, inputs=[query])
        if not q_list:
            return {"ok": False, "error": "embedding_failed"}

        q_vec = np.asarray(q_list[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec)) + 1e-12

        idxs = list(range(len(self._vec_idmap)))
        if scope == "doc" and doc_id is not None:
            did = str(doc_id)
            idxs = [
                i
                for i, m in enumerate(self._vec_idmap)
                if str(m.get("doc_id") or self.node_to_doc_id.get(str(m.get("node_id")), "")) == did
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

        M = self._vec_matrix
        sims: List[float] = []
        for i in idxs:
            v = np.asarray(M[i], dtype=np.float32)
            v_norm = 1.0 if self._vec_normalized else (float(np.linalg.norm(v)) + 1e-12)
            sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
            sim = max(-1.0, min(1.0, sim))
            sims.append(sim)

        order = np.argsort(sims)[::-1]
        k = max(1, int(top_k))

        hits: List[Dict[str, Any]] = []
        for rank in order[:k]:
            global_idx = idxs[int(rank)]
            meta = self._vec_idmap[global_idx]
            nid = str(meta.get("node_id"))
            did = str(meta.get("doc_id") or self.node_to_doc_id.get(nid, ""))
            p_idx = int(meta.get("paragraph_index", 0))

            node = (self.nodes_by_doc.get(did) or {}).get(nid) or {"paragraphs": []}
            text = ""
            pars = node.get("paragraphs", [])
            if 0 <= p_idx < len(pars):
                p = pars[p_idx]
                if isinstance(p, str):
                    text = p
                elif isinstance(p, dict):
                    text = p.get("content", "")
                else:
                    text = str(p)

            neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
            index_set = {p_idx}
            for item in neighbors:
                try:
                    index_set.add(int(item["paragraph_index"]))
                except Exception:
                    continue
            para_indexes = sorted(index_set)

            hits.append(
                {
                    "score": _round_score(sims[int(rank)]),
                    "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                    "text": text,
                    "neighbors": neighbors,
                }
            )

        return {"ok": True, "query": query, "scope": scope, "doc_id": str(doc_id) if doc_id is not None else None, "results": hits}

    def hybrid_search(
        self,
        query: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        top_k: int = 2,
        bm25_weight: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")),
        vector_weight: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.5")),
        top_k_bm25: int = int(os.getenv("HYBRID_TOPK_BM25", "20")),
        top_k_vec: int = int(os.getenv("HYBRID_TOPK_VEC", "20")),
        include_images: bool = True,
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        bm25_res = self.bm25_search(
            query=query,
            scope=scope,
            doc_id=doc_id,
            top_k=top_k_bm25,
            include_images=include_images,
            neighbor_window=neighbor_window,
        )
        vec_res = self.vector_search(
            query=query,
            scope=scope,
            doc_id=doc_id,
            top_k=top_k_vec,
            include_images=include_images,
            embed_api_key=embed_api_key,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            neighbor_window=neighbor_window,
        )

        if not bm25_res.get("ok", False) and not vec_res.get("ok", False):
            return {"ok": False, "error": "both bm25_search and vector_search failed", "query": query, "scope": scope, "doc_id": str(doc_id) if doc_id is not None else None, "results": []}

        bm25_map: Dict[Tuple[str, str, int], float] = {}
        bmax = 0.0
        for r in bm25_res.get("results", []):
            ref = r.get("ref") or {}
            did = str(ref.get("doc_id"))
            nid = str(ref.get("node_id"))
            para_list = ref.get("paragraph_indexes") or []
            if not para_list:
                continue
            p_idx = int(para_list[0])
            score_val = float(r.get("score", 0.0))
            key = (did, nid, p_idx)
            bm25_map[key] = score_val
            bmax = max(bmax, score_val)

        vec_map: Dict[Tuple[str, str, int], float] = {}
        v_scores: List[float] = []
        for r in vec_res.get("results", []):
            ref = r.get("ref") or {}
            did = str(ref.get("doc_id"))
            nid = str(ref.get("node_id"))
            para_list = ref.get("paragraph_indexes") or []
            if not para_list:
                continue
            p_idx = int(para_list[0])
            score_val = float(r.get("score", 0.0))
            key = (did, nid, p_idx)
            vec_map[key] = score_val
            v_scores.append(score_val)

        vmin, vmax = (min(v_scores), max(v_scores)) if v_scores else (0.0, 1.0)

        fused: Dict[Tuple[str, str, int], float] = {}
        keys = set(bm25_map.keys()) | set(vec_map.keys())

        for key in keys:
            b = bm25_map.get(key, 0.0)
            v = vec_map.get(key, 0.0)
            b_norm = (b / bmax) if bmax > 0 else 0.0
            v_norm = ((v - vmin) / (vmax - vmin)) if (vmax - vmin) > 1e-12 else 0.0
            fused[key] = bm25_weight * b_norm + vector_weight * v_norm

        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(top_k))

        hits: List[Dict[str, Any]] = []
        for (did, nid, p_idx), fused_score in ordered[:k]:
            node = (self.nodes_by_doc.get(did) or {}).get(nid) or {"paragraphs": []}
            pars = node.get("paragraphs", [])
            text = ""
            if 0 <= p_idx < len(pars):
                p = pars[p_idx]
                if isinstance(p, str):
                    text = p
                elif isinstance(p, dict):
                    text = p.get("content", "")
                else:
                    text = str(p)

            neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
            index_set = {p_idx}
            for item in neighbors:
                try:
                    index_set.add(int(item["paragraph_index"]))
                except Exception:
                    continue
            para_indexes = sorted(index_set)

            hits.append(
                {
                    "score": _round_score(fused_score),
                    "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                    "text": text,
                    "neighbors": neighbors,
                }
            )

        return {"ok": True, "query": query, "scope": scope, "doc_id": str(doc_id) if doc_id is not None else None, "results": hits}

    def regex_search(
        self,
        pattern: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        top_k: int = 2,
        include_images: bool = True,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        if not pattern:
            return {"ok": False, "error": "empty pattern"}
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            return {"ok": False, "error": f"invalid regex pattern: {exc}"}

        docs = self.par_docs
        if scope == "doc" and doc_id is not None:
            did = str(doc_id)
            docs = [d for d in self.par_docs if d["doc_id"] == did]
            if not docs:
                return {"ok": False, "error": f"no paragraphs under doc '{did}'", "pattern": pattern, "scope": scope, "doc_id": did, "results": []}

        matches: List[Tuple[int, Dict[str, Any]]] = []
        for d in docs:
            found = regex.findall(d["text"])
            if found:
                matches.append((len(found), d))
        matches.sort(key=lambda x: x[0], reverse=True)

        hits: List[Dict[str, Any]] = []
        k = max(1, int(top_k))

        for count, d in matches[:k]:
            did = d["doc_id"]
            nid = d["node_id"]
            p_idx = d["p_idx"]

            neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
            index_set = {p_idx}
            for item in neighbors:
                try:
                    index_set.add(int(item["paragraph_index"]))
                except Exception:
                    continue
            para_indexes = sorted(index_set)

            hits.append(
                {
                    "score": _round_score(float(count)),
                    "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                    "text": d["text"],
                    "neighbors": neighbors,
                }
            )

        return {"ok": True, "pattern": pattern, "scope": scope, "doc_id": str(doc_id) if doc_id is not None else None, "results": hits}

    def _http_rerank_siliconflow(
        self,
        query: str,
        documents: List[str],
        api_key: Optional[str],
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "Qwen/Qwen3-Reranker-8B",
        top_n: int = -1,
        return_documents: bool = True,
        max_chunks_per_doc: int = 1024,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        if not api_key:
            raise RuntimeError("Please set SILICONFLOW_API_KEY (or pass rerank_api_key)")
        url = base_url.rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def semantic_retrieval(
        self,
        query: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        stage1_method: str = "vector",
        top_k1: int = 30,
        top_k2: int = 5,
        stage1_hybrid_topk_bm25: int = 50,
        stage1_hybrid_topk_vec: int = 50,
        include_images: bool = True,
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        rerank_api_key: Optional[str] = None,
        rerank_base_url: str = "https://api.siliconflow.cn/v1",
        rerank_model: str = "Qwen/Qwen3-Reranker-8B",
        neighbor_window: Optional[Tuple[int, int]] = None,
        hybrid_bm25_weight: float = 0.5,
        hybrid_vector_weight: float = 0.5,
    ) -> Dict[str, Any]:
        if not query:
            return {"ok": False, "error": "empty query"}

        k1 = max(1, int(top_k1))
        k2 = max(1, int(top_k2))
        if k1 < k2:
            k1 = k2

        stage1 = (stage1_method or "vector").lower().strip()
        if stage1 not in ("vector", "bm25", "hybrid"):
            stage1 = "vector"

        no_neighbor: Optional[Tuple[int, int]] = None

        candidates_res: Dict[str, Any]
        if stage1 == "bm25":
            candidates_res = self.bm25_search(query=query, scope=scope, doc_id=doc_id, top_k=k1, include_images=False, neighbor_window=no_neighbor)
        elif stage1 == "hybrid":
            b_top = max(k1, int(stage1_hybrid_topk_bm25))
            v_top = max(k1, int(stage1_hybrid_topk_vec))
            candidates_res = self.hybrid_search(
                query=query,
                scope=scope,
                doc_id=doc_id,
                top_k=k1,
                bm25_weight=hybrid_bm25_weight,
                vector_weight=hybrid_vector_weight,
                top_k_bm25=b_top,
                top_k_vec=v_top,
                include_images=False,
                embed_api_key=embed_api_key,
                embed_base_url=embed_base_url,
                embed_model=embed_model,
                neighbor_window=no_neighbor,
            )
        else:
            candidates_res = self.vector_search(
                query=query,
                scope=scope,
                doc_id=doc_id,
                top_k=k1,
                include_images=False,
                embed_api_key=embed_api_key,
                embed_base_url=embed_base_url,
                embed_model=embed_model,
                neighbor_window=no_neighbor,
            )

        if not candidates_res.get("ok", False):
            return {
                "ok": False,
                "error": f"stage1_{stage1}_failed",
                "query": query,
                "scope": scope,
                "doc_id": str(doc_id) if doc_id is not None else None,
                "stage1_method": stage1,
                "top_k1": k1,
                "top_k2": k2,
                "results": [],
            }

        candidates = candidates_res.get("results", []) or []
        if not candidates:
            return {
                "ok": True,
                "query": query,
                "scope": scope,
                "doc_id": str(doc_id) if doc_id is not None else None,
                "stage1_method": stage1,
                "top_k1": k1,
                "top_k2": k2,
                "results": [],
            }

        docs = [str(c.get("text", "")) for c in candidates]
        if not any(docs):
            return {
                "ok": True,
                "query": query,
                "scope": scope,
                "doc_id": str(doc_id) if doc_id is not None else None,
                "stage1_method": stage1,
                "top_k1": k1,
                "top_k2": k2,
                "results": [],
            }

        rkey = rerank_api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("RERANK_API_KEY")

        rerank_ok = True
        rerank_data: Dict[str, Any] = {}
        try:
            rerank_data = self._http_rerank_siliconflow(
                query=query,
                documents=docs,
                api_key=rkey,
                base_url=rerank_base_url,
                model=rerank_model,
                top_n=-1,
                return_documents=True,
                max_chunks_per_doc=1024,
            )
        except Exception as exc:
            rerank_ok = False
            rerank_data = {"error": str(exc)}

        if not rerank_ok:
            hits: List[Dict[str, Any]] = []
            for c in candidates[:k2]:
                ref = c.get("ref") or {}
                did = str(ref.get("doc_id"))
                nid = str(ref.get("node_id"))
                para_list = ref.get("paragraph_indexes") or []
                p_idx = int(para_list[0]) if para_list else 0

                neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
                index_set = {p_idx}
                for nb in neighbors:
                    try:
                        index_set.add(int(nb["paragraph_index"]))
                    except Exception:
                        continue
                para_indexes = sorted(index_set)

                hits.append(
                    {
                        "score": _round_score(float(c.get("score", 0.0))),
                        "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                        "text": c.get("text", ""),
                        "neighbors": neighbors,
                    }
                )

            return {
                "ok": True,
                "query": query,
                "scope": scope,
                "doc_id": str(doc_id) if doc_id is not None else None,
                "stage1_method": stage1,
                "top_k1": k1,
                "top_k2": k2,
                "rerank_ok": False,
                "rerank_error": rerank_data.get("error"),
                "results": hits,
            }

        r_results = rerank_data.get("results", []) or []
        picked = r_results[:k2]

        hits: List[Dict[str, Any]] = []
        for item in picked:
            try:
                idx = int(item.get("index", -1))
            except Exception:
                idx = -1
            if idx < 0 or idx >= len(candidates):
                continue

            base = candidates[idx]
            ref = base.get("ref") or {}
            did = str(ref.get("doc_id"))
            nid = str(ref.get("node_id"))
            para_list = ref.get("paragraph_indexes") or []
            p_idx = int(para_list[0]) if para_list else 0

            neighbors = self._neighbor_context_for(did, nid, p_idx, include_images=include_images, neighbor_window=neighbor_window)
            index_set = {p_idx}
            for nb in neighbors:
                try:
                    index_set.add(int(nb["paragraph_index"]))
                except Exception:
                    continue
            para_indexes = sorted(index_set)

            score = item.get("relevance_score", 0.0)
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0

            hits.append(
                {
                    "score": _round_score(score_f),
                    "ref": {"doc_id": did, "node_id": nid, "paragraph_indexes": para_indexes},
                    "text": base.get("text", ""),
                    "neighbors": neighbors,
                }
            )

        return {
            "ok": True,
            "query": query,
            "scope": scope,
            "doc_id": str(doc_id) if doc_id is not None else None,
            "stage1_method": stage1,
            "top_k1": k1,
            "top_k2": k2,
            "rerank_ok": True,
            "rerank_model": rerank_model,
            "results": hits,
        }


# ------------------------------
# Agent loop helpers
# ------------------------------
def build_system_prompt(doc_index: DocIndex, tool_names: List[str], enable_reasoning: bool = True) -> str:
    overview = doc_index.overview()
    search_tools = [t for t in tool_names if ("search" in (t or "")) or ("retrieval" in (t or ""))]
    search_cmd = f"Use {', '.join(search_tools)}" if search_tools else "Search"

    constraints = [
        f"{search_cmd} to locate relevant nodes based on the directory.",
        "Answer strictly based on the provided corpus; do not fabricate.",
        "The hierarchical structure of documents is represented in the Directory Structure. Parsing errors may cause body text to be mistakenly treated as hierarchical elements (or headings), rendering the heading text inaccessible to search and reading tools. Please make reasonable inferences based on the structure and the content returned by the tool.",
        "Respond in the User's language; align queries with the Directory Structure.",
        "Usually, you need to think step by step and then call tools to locate or read, iterating in this way until you can answer the question.",
        "When calling tools, DO NOT write tool invocations in plain text. Use the structured tool call interface (tool_calls) only.",
    ]

    constraints_block = "\n".join(f"- {c}" for c in constraints)

    return dedent(
        f"""
        You are a documents assistant and will receive one or more documents
        structured as follows:
        `- (doc_id) [node_id] Title | paragraphs=Num | tokens=Num | children=[ID list]`.
        Use this structure and your available tools to answer the user's question.

        ## Guidelines
        {constraints_block}

        ## Directory Structure
        {overview}
        """
    ).strip()


def _neighbor_hint_sentence(doc_index: DocIndex) -> str:
    nw = doc_index.neighbor_window
    if nw is None:
        return ""
    up, down = nw
    return f" Neighbor expansion is enabled: returned hits may include up to {int(up)} paragraph(s) above and {abs(int(down))} paragraph(s) below the matched paragraph."


def make_tools_schema(doc_index: DocIndex, enable_semantic: bool = False) -> List[Dict[str, Any]]:
    nh = _neighbor_hint_sentence(doc_index)

    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_section",
                "description": "Read a specific paragraph range from the specified document node. Returns paragraphs from start_paragraph (inclusive) to end_paragraph (exclusive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "node_id": {"type": "string"},
                        "start_paragraph": {"type": "integer", "minimum": 0},
                        "end_paragraph": {"type": "integer", "minimum": 1},
                    },
                    "required": ["doc_id", "node_id", "start_paragraph", "end_paragraph"],
                },
            },
        }
    ]

    bm25_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    regex_props: Dict[str, Any] = {
        "pattern": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    vector_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }
    hybrid_props: Dict[str, Any] = {
        "query": {"type": "string"},
        "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
        "doc_id": {"type": "string", "default": None},
    }

    tools.extend(
        [
            {
                "type": "function",
                "function": {
                    "name": "bm25_search",
                    "description": "Perform BM25-based text retrieval." + nh,
                    "parameters": {"type": "object", "properties": bm25_props, "required": ["query", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "regex_search",
                    "description": "Search for text patterns using regex." + nh,
                    "parameters": {"type": "object", "properties": regex_props, "required": ["pattern", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Perform embedding-based retrieval using cosine similarity." + nh,
                    "parameters": {"type": "object", "properties": vector_props, "required": ["query", "scope"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hybrid_search",
                    "description": "Fuse BM25 and embedding retrieval results with adjustable weights (internal)." + nh,
                    "parameters": {"type": "object", "properties": hybrid_props, "required": ["query", "scope"]},
                },
            },
        ]
    )

    if enable_semantic:
        semantic_props: Dict[str, Any] = {
            "query": {"type": "string"},
            "scope": {"type": "string", "enum": ["full", "doc"], "default": "full"},
            "doc_id": {"type": "string", "default": None},
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "semantic_retrieval",
                    "description": "Semantic Retrieval: stage-1 recall + rerank internally. Neighbor expansion (if enabled) applies ONLY after reranking on final results." + nh,
                    "parameters": {"type": "object", "properties": semantic_props, "required": ["query", "scope"]},
                },
            }
        )

    return tools


def sanitize_for_vllm(payload: Dict[str, Any], allow_tools: bool = True) -> Dict[str, Any]:
    p = dict(payload)
    for k in ("include_reasoning", "reasoning", "parallel_tool_calls", "response_format", "modalities", "audio", "vision", "metadata"):
        p.pop(k, None)

    if not allow_tools:
        p.pop("tools", None)
        p.pop("tool_choice", None)

    cleaned_msgs: List[Dict[str, Any]] = []
    for m in p.get("messages", []):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        content = m.get("content", "")
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            content = "\n".join(t for t in texts if t)
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        m2: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and allow_tools and m.get("tool_calls"):
            m2["tool_calls"] = m["tool_calls"]
        if role == "tool" and m.get("tool_call_id"):
            m2["tool_call_id"] = m["tool_call_id"]

        cleaned_msgs.append(m2)

    p["messages"] = cleaned_msgs
    return p


def _preview_messages(messages: List[Dict[str, Any]]) -> str:
    try:
        return json.dumps([m for m in messages], ensure_ascii=False)
    except Exception:
        return "<unserializable messages>"


def _preview_tool_calls(tool_calls: Any) -> Any:
    if not tool_calls:
        return None
    out = []
    for tc in tool_calls:
        out.append({"id": tc.get("id"), "name": (tc.get("function") or {}).get("name")})
    return out


def http_chat_completions(
    api_key: Optional[str],
    base_url: Optional[str],
    payload: Dict[str, Any],
    default_headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    logger: Optional["JsonlLogger"] = None,
) -> Dict[str, Any]:
    if not api_key:
        err = RuntimeError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY")
        if logger:
            logger.log("llm_http_error", error=str(err))
        raise err

    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    payload_copy = dict(payload)
    payload_copy["stream"] = False

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if default_headers:
        headers.update(default_headers)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_s = min(90, 1.5 * (2 ** (attempt - 1)))
                time.sleep(wait_s)

            if logger:
                logger.log("llm_http_attempt", attempt=attempt + 1, url=url, model=payload_copy.get("model"))

            resp = requests.post(url, headers=headers, json=payload_copy, timeout=timeout)
            status = resp.status_code

            if status in (429, 500, 502, 503, 504):
                will_retry = attempt < max_retries - 1
                if logger:
                    logger.log("llm_http_error", status_code=status, error=f"HTTP {status}", attempt=attempt + 1, will_retry=will_retry)
                if will_retry:
                    continue
                resp.raise_for_status()

            resp.raise_for_status()
            out = resp.json()
            if logger:
                logger.log("llm_http_success", attempt=attempt + 1, status_code=status)
            return out

        except requests.exceptions.RequestException as exc:
            will_retry = attempt < max_retries - 1
            if logger:
                logger.log("llm_http_error", error=str(exc), attempt=attempt + 1, will_retry=will_retry)
            if will_retry:
                continue
            raise

        except Exception as exc:  # pragma: no cover
            will_retry = attempt < max_retries - 1
            if logger:
                logger.log("llm_http_error", error=str(exc), attempt=attempt + 1, will_retry=will_retry)
            if will_retry:
                continue
            raise

    return {}


def run_agent(
    model: str,
    base_url: Optional[str],
    doc_index: DocIndex,
    user_question: str,
    logger: JsonlLogger,
    max_rounds: int = 12,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    enable_multimodal: bool = False,
    enable_vector: bool = False,
    enable_hybrid: bool = False,
    enable_semantic: bool = False,
    disable_bm25: bool = False,
    disable_regex: bool = False,
    disable_read: bool = False,
    embed_api_key: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
    bm25_topk: int = 1,
    regex_topk: int = 1,
    vector_topk: int = 1,
    hybrid_topk: int = 1,
    hybrid_topk_bm25: int = 30,
    hybrid_topk_vec: int = 30,
    hybrid_bm25_weight: float = 0.5,
    hybrid_vector_weight: float = 0.5,
    semantic_stage1_method: str = "vector",
    semantic_topk1: int = 30,
    semantic_topk2: int = 1,
    semantic_stage1_hybrid_topk_bm25: int = 30,
    semantic_stage1_hybrid_topk_vec: int = 30,
    rerank_api_key: Optional[str] = None,
    rerank_base_url: str = "https://api.siliconflow.cn/v1",
    rerank_model: str = "Qwen/Qwen3-Reranker-8B",
    tool_fallback: bool = True,
    enable_reasoning: bool = True,
) -> str:
    tools = make_tools_schema(doc_index, enable_semantic=enable_semantic)

    if disable_bm25:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "bm25_search"]
    if disable_regex:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "regex_search"]
    if not enable_vector:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "vector_search"]
    if not enable_hybrid:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "hybrid_search"]
    if not enable_semantic:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "semantic_retrieval"]
    if disable_read:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "read_section"]

    tool_names = [(t.get("function") or {}).get("name") for t in tools]
    system_prompt = build_system_prompt(doc_index, tool_names, enable_reasoning=enable_reasoning)

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    prev_msg_count = 1

    do_sanitize = should_sanitize_for_vllm(base_url)

    effective_neighbor_window: Optional[Tuple[int, int]] = neighbor_window if neighbor_window is not None else doc_index.neighbor_window

    for round_id in range(1, max_rounds + 1):
        req_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "stream": False,
            "include_reasoning": bool(enable_reasoning),
        }

        logger.log("llm_request", round=round_id, base_url=base_url, context_delta_preview=_preview_messages(messages[prev_msg_count:]))
        prev_msg_count = len(messages)

        payload_to_send = sanitize_for_vllm(req_payload, allow_tools=True) if do_sanitize else req_payload

        try:
            resp = http_chat_completions(api_key=api_key, base_url=base_url, payload=payload_to_send, default_headers=default_headers, logger=logger)
        except Exception as exc:
            logger.log("llm_http_error", error=str(exc), round=round_id)
            resp = {}

        msg = (resp.get("choices") or [{}])[0].get("message", {})  # type: ignore

        reasoning_content = None
        if enable_reasoning:
            for rf in ["reasoning", "reasoning_content", "thinking", "internal_monologue"]:
                if msg.get(rf) is not None:
                    reasoning_content = msg.get(rf)
                    break

        tool_calls = msg.get("tool_calls")
        recovered_from_text = False
        recovered_meta: Dict[str, Any] = {}

        content_str = msg.get("content") or ""
        reasoning_str = reasoning_content or ""

        if not tool_calls and tool_fallback:
            rec = fallback_tool_calls_from_text(content_str) or fallback_tool_calls_from_text(reasoning_str)
            if rec:
                tool_calls, recovered_meta = rec
                recovered_from_text = True
                logger.log(
                    "tool_calls_recovered_from_text",
                    round=round_id,
                    recovered=_preview_tool_calls(tool_calls),
                    recovered_kind=recovered_meta.get("kind"),
                )

        assistant_entry: Dict[str, Any] = {"role": "assistant"}

        content_for_history = msg.get("content")
        if isinstance(content_for_history, str) and recovered_from_text:
            content_for_history = strip_function_calls_block_any(content_for_history)
            if recovered_meta.get("kind") == "inline_json" and recovered_meta.get("spans"):
                content_for_history = strip_inline_tool_calls(content_for_history, recovered_meta["spans"])

        if content_for_history is None:
            content_for_history = ""
        assistant_entry["content"] = content_for_history

        if enable_reasoning and reasoning_content:
            assistant_entry["reasoning"] = reasoning_content

        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls

        messages.append(assistant_entry)

        logger.log(
            "llm_response",
            round=round_id,
            content=msg.get("content"),
            reasoning_content=reasoning_content if enable_reasoning else None,
            tool_calls=_preview_tool_calls(tool_calls),
            context_delta_preview=_preview_messages(messages[prev_msg_count:]) if tool_calls else None,
        )

        if tool_calls:
            prev_msg_count = len(messages)

        if not tool_calls:
            final_answer = (msg.get("content") or "").strip()
            if final_answer:
                logger.log("final_answer", answer=final_answer, context_delta_preview=_preview_messages(messages[prev_msg_count:]))
                return final_answer

            if enable_reasoning and (reasoning_content is not None) and str(reasoning_content).strip():
                logger.log("llm_thinking_only", round=round_id, reasoning_preview=str(reasoning_content)[:2000])
            else:
                logger.log("llm_empty_message", round=round_id)
            continue

        for tc in tool_calls or []:
            tool_name = (tc.get("function") or {}).get("name")
            try:
                args_raw = (tc.get("function") or {}).get("arguments")
                args = args_raw if isinstance(args_raw, dict) else json.loads(args_raw or "{}")
            except Exception as exc:
                logger.log("tool_args_parse_error", tool=tool_name, raw=str(args_raw), error=str(exc))
                args = {}

            logger.log("tool_call", tool=tool_name, args=args, tool_call_id=tc.get("id"))

            try:
                if tool_name == "read_section":
                    out = doc_index.read_section(
                        doc_id=args.get("doc_id"),
                        node_id=args.get("node_id"),
                        start_paragraph=int(args.get("start_paragraph", 0)),
                        end_paragraph=int(args.get("end_paragraph", -1)),
                        include_images=enable_multimodal,
                    )
                elif tool_name == "bm25_search":
                    out = doc_index.bm25_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=int(bm25_topk),
                        include_images=enable_multimodal,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "regex_search":
                    out = doc_index.regex_search(
                        pattern=args.get("pattern", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=int(regex_topk),
                        include_images=enable_multimodal,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "vector_search":
                    out = doc_index.vector_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=int(vector_topk),
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "hybrid_search":
                    out = doc_index.hybrid_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=int(hybrid_topk),
                        bm25_weight=float(hybrid_bm25_weight),
                        vector_weight=float(hybrid_vector_weight),
                        top_k_bm25=int(hybrid_topk_bm25),
                        top_k_vec=int(hybrid_topk_vec),
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "semantic_retrieval":
                    out = doc_index.semantic_retrieval(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        stage1_method=str(semantic_stage1_method),
                        top_k1=int(semantic_topk1),
                        top_k2=int(semantic_topk2),
                        stage1_hybrid_topk_bm25=int(semantic_stage1_hybrid_topk_bm25),
                        stage1_hybrid_topk_vec=int(semantic_stage1_hybrid_topk_vec),
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        rerank_api_key=rerank_api_key,
                        rerank_base_url=rerank_base_url,
                        rerank_model=rerank_model,
                        neighbor_window=effective_neighbor_window,
                        hybrid_bm25_weight=float(hybrid_bm25_weight),
                        hybrid_vector_weight=float(hybrid_vector_weight),
                    )
                else:
                    out = {"ok": False, "error": f"Tool '{tool_name}' not implemented"}

                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(out, ensure_ascii=False)})

                if (
                    enable_multimodal
                    and tool_name in ("bm25_search", "regex_search", "vector_search", "hybrid_search", "semantic_retrieval")
                    and isinstance(out, dict)
                ):
                    seen_keys = set()
                    mm_items: List[Dict[str, Any]] = []
                    for r in out.get("results", []):
                        neighbors = r.get("neighbors") or []
                        ref = r.get("ref") or {}
                        nid = str(ref.get("node_id"))
                        for item in neighbors:
                            if isinstance(item, dict):
                                item_type = item.get("type") or "text"
                                par_idx = int(item.get("paragraph_index", -1))
                                key = (nid, par_idx, item_type)
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                tagged = dict(item)
                                tagged["paragraph_index"] = par_idx
                                mm_items.append(tagged)
                    if mm_items:
                        messages.append({"role": "user", "content": mm_items})

                logger.log(
                    "tool_result",
                    tool=tool_name,
                    ok=bool(out.get("ok", True)) if isinstance(out, dict) else True,
                    result=out,
                    context_delta_preview=_preview_messages(messages[prev_msg_count:]),
                )
                prev_msg_count = len(messages)

            except Exception as exc:
                err = {"ok": False, "error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(err, ensure_ascii=False)})
                logger.log(
                    "tool_result",
                    tool=tool_name,
                    ok=False,
                    error=str(exc),
                    result=err,
                    context_delta_preview=_preview_messages(messages[prev_msg_count:]),
                )
                prev_msg_count = len(messages)

    logger.log("max_rounds_reached", max_rounds=max_rounds)
    return "(Reached maximum rounds, no final answer generated)"


def load_corpus(paths: List[str], neighbor_window: Optional[Tuple[int, int]]) -> DocIndex:
    all_nodes: List[Dict[str, Any]] = []
    mats: List[np.ndarray] = []
    idmaps: List[Dict[str, Any]] = []
    models: List[str] = []
    norms: List[bool] = []

    for idx, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = data.get("nodes") or []
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("corpus JSON must contain a non-empty 'nodes' list")

        doc_id = str(idx + 1)
        for n in nodes:
            n["doc_id"] = doc_id
        all_nodes.extend(nodes)

        vs = data.get("vector_store")
        if isinstance(vs, dict):
            mp = vs.get("matrix_path")
            ip = vs.get("id_map_path")
            if mp and ip and Path(mp).exists() and Path(ip).exists():
                arr = np.load(mp, mmap_mode="r").astype(np.float32)
                with open(ip, "r", encoding="utf-8") as f_id:
                    im = json.load(f_id) or []
                for entry in im:
                    if entry.get("doc_id") is None:
                        entry["doc_id"] = doc_id
                mats.append(arr)
                idmaps.extend(im)
                models.append(str(vs.get("model_name")))
                norms.append(bool(vs.get("normalized", False)))

    di = DocIndex(all_nodes, neighbor_window=neighbor_window)

    if mats:
        di._vec_matrix = np.concatenate(mats, axis=0)
        di._vec_idmap = idmaps
        di._vec_model_name = models[0] if models else None
        di._vec_normalized = all(norms) if norms else False

    return di


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True, nargs="+", help="one or more corpus JSON file paths")
    parser.add_argument("--question", required=True, help="user question")
    parser.add_argument("--log", default="run_log.jsonl", help="jsonl log output path")
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-multimodal", action="store_true", help="enable multimodal support (images + text)")

    parser.add_argument("--tool-fallback", dest="tool_fallback", action="store_true", default=True, help="Enable tool call fallback parsing (default: ON)")
    parser.add_argument("--no-tool-fallback", dest="tool_fallback", action="store_false", help="Disable tool call fallback parsing")

    parser.add_argument("--enable-reasoning", dest="enable_reasoning", action="store_true", default=True, help="Enable reasoning/thinking (default: ON)")
    parser.add_argument("--disable-reasoning", dest="enable_reasoning", action="store_false", help="Disable reasoning/thinking")

    parser.add_argument("--embedding-model", type=str, default=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    parser.add_argument("--embed-base-url", type=str, default=os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--embed-api-key", type=str, default=os.getenv("EMBED_API_KEY", ""))

    parser.add_argument("--enable-vector", action="store_true")
    parser.add_argument("--enable-hybrid", action="store_true")
    parser.add_argument("--enable-semantic", action="store_true")
    parser.add_argument("--disable-bm25", action="store_true")
    parser.add_argument("--disable-regex", action="store_true")
    parser.add_argument("--disable-read", action="store_true")

    parser.add_argument("--bm25-topk", type=int, default=int(os.getenv("BM25_TOPK", "1")))
    parser.add_argument("--regex-topk", type=int, default=int(os.getenv("REGEX_TOPK", "1")))
    parser.add_argument("--vector-topk", type=int, default=int(os.getenv("VECTOR_TOPK", "1")))
    parser.add_argument("--hybrid-topk", type=int, default=int(os.getenv("HYBRID_TOPK", "1")))

    parser.add_argument("--hybrid-topk-bm25", type=int, default=int(os.getenv("HYBRID_TOPK_BM25", "30")))
    parser.add_argument("--hybrid-topk-vec", type=int, default=int(os.getenv("HYBRID_TOPK_VEC", "30")))
    parser.add_argument("--hybrid-bm25-weight", type=float, default=float(os.getenv("HYBRID_BM25_WEIGHT", "0.5")))
    parser.add_argument("--hybrid-vector-weight", type=float, default=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.5")))

    parser.add_argument("--semantic-stage1", type=str, default=os.getenv("SEMANTIC_STAGE1", "vector"), choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--semantic-topk1", type=int, default=int(os.getenv("SEMANTIC_TOPK1", "30")))
    parser.add_argument("--semantic-topk2", type=int, default=int(os.getenv("SEMANTIC_TOPK2", "2")))
    parser.add_argument("--semantic-stage1-hybrid-topk-bm25", type=int, default=int(os.getenv("SEMANTIC_STAGE1_HYBRID_TOPK_BM25", "30")))
    parser.add_argument("--semantic-stage1-hybrid-topk-vec", type=int, default=int(os.getenv("SEMANTIC_STAGE1_HYBRID_TOPK_VEC", "30")))

    parser.add_argument("--rerank-api-key", type=str, default=os.getenv("SILICONFLOW_API_KEY", "") or os.getenv("RERANK_API_KEY", ""))
    parser.add_argument("--rerank-base-url", type=str, default=os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1"))
    parser.add_argument("--rerank-model", type=str, default=os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B"))

    parser.add_argument(
        "--neighbor-window",
        type=str,
        default=os.getenv("NEIGHBOR_WINDOW", "1,-1"),
        help="Neighbor window as 'up,down' (up>=0, down<=0). '0,0' disables neighbors.",
    )

    args = parser.parse_args()

    try:
        parts = [p.strip() for p in str(args.neighbor_window).split(",")]
        if len(parts) != 2:
            raise ValueError("--neighbor-window must be in 'up,down' form, e.g. '1,-1' or '0,0'")
        neighbor_window_arg = _normalize_neighbor_window((int(parts[0]), int(parts[1])))
    except Exception as exc:
        raise SystemExit(f"Invalid --neighbor-window: {exc}")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY or OPENROUTER_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")

    default_headers: Optional[Dict[str, str]] = None
    if base_url and "openrouter.ai" in base_url:
        default_headers = {"HTTP-Referer": os.getenv("ORIGIN", "http://localhost"), "X-Title": os.getenv("APP_NAME", "doc-agent")}

    model = os.getenv("OPENAI_MODEL") or ("openai/gpt-4o" if (base_url and "openrouter.ai" in base_url) else "gpt-4o")

    logger = JsonlLogger(args.log)
    doc_index = load_corpus(args.doc, neighbor_window=neighbor_window_arg)

    answer = run_agent(
        model=model,
        base_url=base_url,
        doc_index=doc_index,
        user_question=args.question,
        logger=logger,
        max_rounds=args.max_rounds,
        temperature=args.temperature,
        api_key=api_key,
        default_headers=default_headers,
        enable_multimodal=args.enable_multimodal,
        enable_vector=args.enable_vector,
        enable_hybrid=args.enable_hybrid,
        enable_semantic=args.enable_semantic,
        disable_bm25=args.disable_bm25,
        disable_regex=args.disable_regex,
        disable_read=args.disable_read,
        embed_api_key=args.embed_api_key,
        embed_base_url=args.embed_base_url,
        embedding_model=args.embedding_model,
        neighbor_window=neighbor_window_arg,
        bm25_topk=args.bm25_topk,
        regex_topk=args.regex_topk,
        vector_topk=args.vector_topk,
        hybrid_topk=args.hybrid_topk,
        hybrid_topk_bm25=args.hybrid_topk_bm25,
        hybrid_topk_vec=args.hybrid_topk_vec,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_vector_weight=args.hybrid_vector_weight,
        semantic_stage1_method=args.semantic_stage1,
        semantic_topk1=args.semantic_topk1,
        semantic_topk2=args.semantic_topk2,
        semantic_stage1_hybrid_topk_bm25=args.semantic_stage1_hybrid_topk_bm25,
        semantic_stage1_hybrid_topk_vec=args.semantic_stage1_hybrid_topk_vec,
        rerank_api_key=args.rerank_api_key,
        rerank_base_url=args.rerank_base_url,
        rerank_model=args.rerank_model,
        tool_fallback=args.tool_fallback,
        enable_reasoning=args.enable_reasoning,
    )

    print("\n==== Final Answer ====")
    print(answer)
