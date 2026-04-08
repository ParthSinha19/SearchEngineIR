import logging
import math
from typing import Dict, Tuple

from searcher import RunDict

logger = logging.getLogger(__name__)

# Qrels type: query_id -> {doc_id: relevance_grade (int)}
QrelsDict = Dict[str, Dict[str, int]]


def compute_ndcg(
    run: RunDict,
    qrels: QrelsDict,
    cutoff: int = 10,
) -> float:
    
    ndcg_scores = []

    for qid, ranked_list in run.items():
        if qid not in qrels:
            # No judgments for this query — skip to avoid division by zero
            continue

        q_rels = qrels[qid]  # {doc_id: grade}

        # --- Compute DCG@cutoff for this query ---
        dcg = 0.0
        for rank_0, (docid, _) in enumerate(ranked_list[:cutoff]):
            rank = rank_0 + 1
            rel = q_rels.get(docid, 0)
            # Standard DCG formula with graded relevance
            dcg += (2 ** rel - 1) / math.log2(rank + 1)

        # --- Compute IDCG@cutoff (ideal ranking) ---
        # Sort all judged relevant docs by grade descending
        ideal_grades = sorted(q_rels.values(), reverse=True)[:cutoff]
        idcg = 0.0
        for rank_0, rel in enumerate(ideal_grades):
            rank = rank_0 + 1
            idcg += (2 ** rel - 1) / math.log2(rank + 1)

        if idcg == 0:
            # Query has no relevant documents — contributes 0 to average
            ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(dcg / idcg)

    if not ndcg_scores:
        return 0.0
    return sum(ndcg_scores) / len(ndcg_scores)


def compute_recall(
    run: RunDict,
    qrels: QrelsDict,
    cutoff: int = 100,
) -> float:
    
    recall_scores = []

    for qid, ranked_list in run.items():
        if qid not in qrels:
            continue

        # Count all relevant docs (grade >= 1) for this query
        relevant_docs = {docid for docid, grade in qrels[qid].items() if grade >= 1}

        if not relevant_docs:
            continue  # Undefined recall for queries with no relevant docs

        retrieved_at_k = {docid for docid, _ in ranked_list[:cutoff]}
        hits = len(relevant_docs & retrieved_at_k)
        recall_scores.append(hits / len(relevant_docs))

    if not recall_scores:
        return 0.0
    return sum(recall_scores) / len(recall_scores)


def compute_judged_at_k(
    run: RunDict,
    qrels: QrelsDict,
    k: int = 10,
) -> float:
    
    judged_fractions = []

    for qid, ranked_list in run.items():
        if qid not in qrels:
            continue

        judged_docids = set(qrels[qid].keys())
        top_k = [docid for docid, _ in ranked_list[:k]]

        if not top_k:
            continue

        judged_count = sum(1 for d in top_k if d in judged_docids)
        judged_fractions.append(judged_count / len(top_k))

    if not judged_fractions:
        return 0.0
    return sum(judged_fractions) / len(judged_fractions)


def evaluate_run(
    run: RunDict,
    qrels: QrelsDict,
    run_name: str = "unnamed",
) -> Dict[str, float]:

    metrics = {
        "nDCG@10":    compute_ndcg(run, qrels, cutoff=10),
        "Recall@100": compute_recall(run, qrels, cutoff=100),
        "judged@10":  compute_judged_at_k(run, qrels, k=10),
    }

    # Use print() in addition to logger — Colab suppresses logger.info from
    # submodules called via !python main.py but always shows print() stdout
    output = (
        f"\n{'='*50}\n"
        f"  Run: {run_name}\n"
        f"  nDCG@10    : {metrics['nDCG@10']:.4f}\n"
        f"  Recall@100 : {metrics['Recall@100']:.4f}\n"
        f"  judged@10  : {metrics['judged@10']:.4f}\n"
        f"{'='*50}"
    )
    print(output)
    logger.info(output)

    return metrics