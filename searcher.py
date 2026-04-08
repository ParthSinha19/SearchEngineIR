import logging
import os
from typing import Dict, List, Tuple

# Prevents a Rust/Java thread deadlock between HuggingFace's fast tokenizers
# (which use Rust-based parallelism) and Lucene's JVM threads. Must be set
# before any pyserini or transformers import.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger(__name__)

RankedList = List[Tuple[str, float]]
RunDict    = Dict[str, RankedList]


# ---------------------------------------------------------------------------
# 1. BM25 Baseline Searcher
# ---------------------------------------------------------------------------

class BM25Searcher:

    def __init__(
            self,
            index_path: str,
            k1: float = 0.9,
            b: float  = 0.4,
            top_k: int = 100,
    ):
        self.top_k = top_k
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)
        logger.info(f"BM25Searcher ready | k1={k1} | b={b} | index={index_path}")

    def search(self, queries: List[Tuple[str, str]]) -> RunDict:
        run = {}
        for qid, qtext in tqdm(queries, desc="BM25 Search"):
            hits = self.searcher.search(qtext, k=self.top_k)
            run[qid] = [(h.docid, h.score) for h in hits]
        logger.info(f"BM25 search complete: {len(run)} queries")
        return run


# ---------------------------------------------------------------------------
# 2. RM3 Pseudo-Relevance Feedback Searcher
# ---------------------------------------------------------------------------

class RM3Searcher:

    def __init__(
            self,
            index_path: str,
            k1: float = 0.9,
            b: float  = 0.4,
            top_k: int = 100,
            fb_docs:    int   = 10,
            fb_terms:   int   = 10,
            original_query_weight: float = 0.5,
    ):
        self.top_k = top_k
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)
        self.searcher.set_rm3(
            fb_docs=fb_docs,
            fb_terms=fb_terms,
            original_query_weight=original_query_weight,
        )
        logger.info(
            f"RM3Searcher ready | fb_docs={fb_docs} | fb_terms={fb_terms} "
            f"| α={original_query_weight}"
        )

    def search(self, queries: List[Tuple[str, str]]) -> RunDict:
        run = {}
        for qid, qtext in tqdm(queries, desc="RM3 Search"):
            hits = self.searcher.search(qtext, k=self.top_k)
            run[qid] = [(h.docid, h.score) for h in hits]
        logger.info(f"RM3 search complete: {len(run)} queries")
        return run


# ---------------------------------------------------------------------------
# 3. Dense Retrieval (Bi-Encoder / DPR via FAISS)
# ---------------------------------------------------------------------------

class DenseSearcher:

    def __init__(
            self,
            faiss_index_dir: str,
            query_encoder_name: str = "castorini/tct_colbert-v2-hnp-msmarco",
            top_k: int = 100,
            device: str = "cpu",
    ):
        from pyserini.search.faiss import FaissSearcher

        self.searcher = FaissSearcher(faiss_index_dir, query_encoder_name)
        self.top_k = top_k
        logger.info(f"DenseSearcher ready | model={query_encoder_name} | device={device}")

    def search(self, queries: List[Tuple[str, str]]) -> RunDict:
        run = {}
        for qid, qtext in tqdm(queries, desc="Dense Encoding & Search"):
            hits = self.searcher.search(qtext, k=self.top_k)
            run[qid] = [(h.docid, h.score) for h in hits]
        logger.info(f"Dense search complete: {len(run)} queries")
        return run


# ---------------------------------------------------------------------------
# 4. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    run_a: RunDict,
    run_b: RunDict,
    k: int = 60,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
) -> RunDict:
    all_qids = set(run_a.keys()) | set(run_b.keys())
    fused_run = {}

    for qid in all_qids:
        scores: Dict[str, float] = {}

        for rank_0, (docid, _) in enumerate(run_a.get(qid, [])):
            rank = rank_0 + 1
            scores[docid] = scores.get(docid, 0.0) + weight_a / (k + rank)

        for rank_0, (docid, _) in enumerate(run_b.get(qid, [])):
            rank = rank_0 + 1
            scores[docid] = scores.get(docid, 0.0) + weight_b / (k + rank)

        fused_run[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"RRF fusion complete: {len(fused_run)} queries fused")
    return fused_run