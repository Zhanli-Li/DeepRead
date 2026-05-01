from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .bm25_search import bm25_search
from .hybrid_search import hybrid_search
from .read_section import read_section
from .regex_search import regex_search
from .semantic_retrieval import semantic_retrieval
from .utils import _normalize_neighbor_window, count_model_tokens, simple_tokenize
from .vector_search import vector_search


class DocIndex:
    def __init__(self, nodes: List[Dict[str, Any]], neighbor_window: Optional[Tuple[int, int]]) -> None:
        self.neighbor_window: Optional[Tuple[int, int]] = _normalize_neighbor_window(neighbor_window)

        self.nodes_by_doc: Dict[str, Dict[str, Any]] = {}

        for node in nodes:
            node_id = str(node["id"])
            doc_id = str(node.get("doc_id", ""))
            node.setdefault("title", node_id)
            node.setdefault("paragraphs", [])
            node.setdefault("children", [])

            text_parts: List[str] = []
            for paragraph in node["paragraphs"]:
                if isinstance(paragraph, str):
                    text_parts.append(paragraph)
                elif isinstance(paragraph, dict):
                    content = paragraph.get("content", "")
                    if content:
                        text_parts.append(content)
                else:
                    text_parts.append(str(paragraph))
            full_text = "\n".join(text_parts) if text_parts else ""
            node["_tokens"] = simple_tokenize(full_text)
            node["_model_token_count"] = count_model_tokens(full_text)

            self.nodes_by_doc.setdefault(doc_id, {})
            self.nodes_by_doc[doc_id][node_id] = node

        self.par_docs: List[Dict[str, Any]] = []
        for doc_id, doc_nodes in self.nodes_by_doc.items():
            for node_id, node in doc_nodes.items():
                for paragraph_index, paragraph in enumerate(node["paragraphs"]):
                    if isinstance(paragraph, str):
                        text = paragraph
                    elif isinstance(paragraph, dict):
                        text = paragraph.get("content", "")
                    else:
                        text = str(paragraph)
                    if text:
                        tokens = simple_tokenize(text)
                        self.par_docs.append(
                            {
                                "doc_id": str(doc_id),
                                "node_id": str(node_id),
                                "p_idx": paragraph_index,
                                "text": text,
                                "tokens": tokens,
                                "len": len(tokens),
                            }
                        )

        self.N = len(self.par_docs) if self.par_docs else 1
        self.avgdl = sum(d["len"] for d in self.par_docs) / self.N

        df: Dict[str, int] = {}
        for doc in self.par_docs:
            for term in set(doc["tokens"]):
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
        for doc_id, doc_nodes in self.nodes_by_doc.items():
            for node_id, node in doc_nodes.items():
                lines.append(
                    f"- (doc_id={doc_id}) "
                    f"[{node_id}] {node['title']} | paragraphs={len(node['paragraphs'])} | "
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
        paragraph_count = len(node.get("paragraphs", []))

        if window is None or paragraph_count == 0:
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

        if down > 0 and paragraph_index + 1 < paragraph_count:
            start = paragraph_index + 1
            end = min(paragraph_count, paragraph_index + 1 + down)
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
        return read_section(self, doc_id, node_id, start_paragraph, end_paragraph, include_images)

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
        return bm25_search(self, query, scope, doc_id, top_k, k1, b, include_images, neighbor_window)

    def regex_search(
        self,
        pattern: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        top_k: int = 2,
        include_images: bool = True,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        return regex_search(self, pattern, scope, doc_id, top_k, include_images, neighbor_window)

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
        return vector_search(
            self,
            query,
            scope,
            doc_id,
            top_k,
            include_images,
            embed_api_key,
            embed_base_url,
            embed_model,
            neighbor_window,
        )

    def hybrid_search(
        self,
        query: str,
        scope: str = "full",
        doc_id: Optional[str] = None,
        top_k: int = 2,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        top_k_bm25: int = 20,
        top_k_vec: int = 20,
        include_images: bool = True,
        embed_api_key: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        neighbor_window: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        return hybrid_search(
            self,
            query,
            scope,
            doc_id,
            top_k,
            bm25_weight,
            vector_weight,
            top_k_bm25,
            top_k_vec,
            include_images,
            embed_api_key,
            embed_base_url,
            embed_model,
            neighbor_window,
        )

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
        return semantic_retrieval(
            self,
            query,
            scope,
            doc_id,
            stage1_method,
            top_k1,
            top_k2,
            stage1_hybrid_topk_bm25,
            stage1_hybrid_topk_vec,
            include_images,
            embed_api_key,
            embed_base_url,
            embed_model,
            rerank_api_key,
            rerank_base_url,
            rerank_model,
            neighbor_window,
            hybrid_bm25_weight,
            hybrid_vector_weight,
        )
