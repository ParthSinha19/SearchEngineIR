import logging
import time
import json
from pathlib import Path
import torch
from typing import Dict

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ── Project modules ──────────────────────────────────────────────────────────
from data_loader import (
    download_and_load_dataset,
    build_pyserini_jsonl_corpus,
    get_query_list,
)
from indexer import BM25Indexer
from searcher import BM25Searcher, RM3Searcher, DenseSearcher, reciprocal_rank_fusion
from evaluator import evaluate_run

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Config ────────────────────────────────────────────────────────────────────

CONFIG = {
    # Dataset
    "dataset":     "scifact",   # Change to 'trec-covid' for second benchmark
    "split":       "test",
    "data_dir":    "./data",

    # Index paths
    "corpus_dir":  "./data/scifact_pyserini",
    "index_dir":   "./indexes/scifact_bm25",
    "faiss_dir":   "./indexes/scifact_faiss",

    # BM25 hyperparameters (tuned for biomedical/scientific text)
    "bm25_k1":          0.9,
    "bm25_b":           0.4,
    "bm25_title_boost": 3,

    # RM3 hyperparameters
    "rm3_fb_docs":   10,
    "rm3_fb_terms":  10,
    "rm3_alpha":     0.5,

    # Dense retrieval — CPU is safer across Colab CUDA/PyTorch version combos;
    # encoder only runs on 300 queries so the cost is ~2-3s either way
    "dense_encoder": "castorini/tct_colbert-v2-hnp-msmarco",
    "dense_device":  "cpu",

    # RRF fusion
    "rrf_k": 60,

    # Retrieval depth
    "top_k": 100,
}


def warm_up_cuda() -> None:
    """
    Force CUDA context to initialize before any Pyserini/HuggingFace code runs.
    Without this, the first CUDA call inside a subprocess or lazy loader can
    silently hang waiting for the driver to finish initializing.
    """
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        torch.zeros(1).cuda()
        logger.info("CUDA context warmed up.")
    else:
        logger.info("No CUDA device found — running fully on CPU.")


def timed(fn, *args, label="", **kwargs):
    """Run fn(*args, **kwargs), log elapsed time, return result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info(f"[TIMING] {label}: {elapsed:.2f}s")
    return result

# ── Optional Interactive Query Mode ───────────────────────────────────────
def interactive_query_mode(
    bm25_searcher: BM25Searcher,
    corpus: dict,
    top_n: int = 5,
) -> None:
    print("\n" + "="*60)
    print("  INTERACTIVE QUERY MODE (BM25)")
    print("  Type a query to retrieve passages. Leave blank to exit.")
    print("="*60)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError is raised in non-interactive environments (e.g. piped input)
            # KeyboardInterrupt = Ctrl+C
            print("\nExiting interactive mode.")
            break

        if not query:
            print("Exiting interactive mode.")
            break

        from data_loader import preprocess_text
        query = preprocess_text(query)

        hits = bm25_searcher.searcher.search(query, k=top_n)

        if not hits:
            print("No results found.")
            continue

        print(f"\nTop {top_n} results for: '{query}'\n" + "-"*60)
        for i, hit in enumerate(hits, start=1):
            doc = corpus.get(hit.docid, {})
            title = doc.get("title", "[no title]")
            text  = doc.get("text",  "[no text]")
            # Truncate long abstracts for readability
            snippet = text[:300] + "..." if len(text) > 300 else text
            print(f"\n[{i}] Score: {hit.score:.4f} | ID: {hit.docid}")
            print(f"    Title:   {title}")
            print(f"    Snippet: {snippet}")
        print("-"*60)


def main():
    logger.info(f"Starting experiment pipeline. Config:\n{json.dumps(CONFIG, indent=2)}")

    # Warm up CUDA before anything else touches it
    warm_up_cuda()

    # ── 1. Data Loading ───────────────────────────────────────────────────────
    logger.info("=== PHASE 1: Data Loading ===")
    corpus, queries, qrels = timed(
        download_and_load_dataset,
        CONFIG["dataset"], CONFIG["data_dir"], CONFIG["split"],
        label="dataset_load",
    )

    corpus_dir = timed(
        build_pyserini_jsonl_corpus,
        corpus, CONFIG["corpus_dir"],
        label="corpus_jsonl_build",
    )

    query_list = get_query_list(queries)
    sample_qid = list(qrels.keys())[0]
    sample_run_qid = query_list[0][0]
    print(f"qrels key sample: '{sample_qid}' (type: {type(sample_qid).__name__})")
    print(f"query_list key sample: '{sample_run_qid}' (type: {type(sample_run_qid).__name__})")
    logger.info(f"Query list ready: {len(query_list)} queries")

    # ── 2. Indexing ───────────────────────────────────────────────────────────
    logger.info("=== PHASE 2: Lucene Indexing ===")
    indexer = BM25Indexer(
        input_dir=corpus_dir,
        index_dir=CONFIG["index_dir"],
        k1=CONFIG["bm25_k1"],
        b=CONFIG["bm25_b"],
        title_boost=CONFIG["bm25_title_boost"],
    )
    timed(lambda: indexer.build_index(), label="lucene_index_build")

    # ── 3. Ablation Experiments ───────────────────────────────────────────────
    all_metrics = {}

    # ── Experiment 1: BM25 Baseline ──────────────────────────────────────────
    logger.info("=== EXPERIMENT 1: BM25 Baseline ===")
    bm25_searcher = BM25Searcher(
        index_path=CONFIG["index_dir"],  # Fixed from index_path
        k1=CONFIG["bm25_k1"],
        b=CONFIG["bm25_b"],
        top_k=CONFIG["top_k"],
    )
    bm25_run = timed(bm25_searcher.search, query_list, label="bm25_search")
    all_metrics["BM25"] = evaluate_run(bm25_run, qrels, run_name="BM25 Baseline")

    # ── Experiment 2: BM25 + RM3 PRF ─────────────────────────────────────────
    logger.info("=== EXPERIMENT 2: BM25 + RM3 ===")
    rm3_searcher = RM3Searcher(
        index_path=CONFIG["index_dir"],  # Fixed from index_path
        k1=CONFIG["bm25_k1"],
        b=CONFIG["bm25_b"],
        top_k=CONFIG["top_k"],
        fb_docs=CONFIG["rm3_fb_docs"],
        fb_terms=CONFIG["rm3_fb_terms"],
        original_query_weight=CONFIG["rm3_alpha"],
    )
    rm3_run = timed(rm3_searcher.search, query_list, label="rm3_search")
    all_metrics["BM25+RM3"] = evaluate_run(rm3_run, qrels, run_name="BM25 + RM3")

    # ── Experiment 3: Dense Retrieval (Zero-shot) ─────────────────────────────
    logger.info("=== EXPERIMENT 3: Dense Retrieval ===")
    faiss_index_file = Path(CONFIG["faiss_dir"]) / "index"
    faiss_docid_file = Path(CONFIG["faiss_dir"]) / "docid"

    if faiss_index_file.exists() and faiss_docid_file.exists():
        logger.info(f"FAISS index found at {CONFIG['faiss_dir']} — loading DenseSearcher...")
        dense_searcher = DenseSearcher(
            faiss_index_dir=CONFIG["faiss_dir"],
            query_encoder_name=CONFIG["dense_encoder"],
            top_k=CONFIG["top_k"],
            device=CONFIG["dense_device"],
        )
        dense_run = timed(dense_searcher.search, query_list, label="dense_search")
        all_metrics["Dense"] = evaluate_run(dense_run, qrels, run_name="Dense (Zero-shot)")
    else:
        logger.warning(
            f"FAISS index incomplete at {CONFIG['faiss_dir']}. "
            f"Found: {list(Path(CONFIG['faiss_dir']).iterdir()) if Path(CONFIG['faiss_dir']).exists() else 'directory missing'}. "
            "Run the pyserini.encode step first. Skipping Experiments 3 & 4."
        )
        dense_run = None

    # ── Experiment 4: Hybrid RRF Fusion ──────────────────────────────────────
    if dense_run is not None:
        logger.info("=== EXPERIMENT 4: Hybrid RRF Fusion ===")
        rrf_run = timed(
            lambda: reciprocal_rank_fusion(bm25_run, dense_run, k=CONFIG["rrf_k"]),
            label="rrf_fusion",
        )
        all_metrics["RRF(BM25+Dense)"] = evaluate_run(
            rrf_run, qrels, run_name=f"RRF k={CONFIG['rrf_k']} (BM25 + Dense)"
        )
    else:
        logger.warning("Skipping RRF fusion (no dense run available).")

    # ── 4. Final Summary Table ────────────────────────────────────────────────
    header = f"{'Run':<22} {'nDCG@10':>10} {'Recall@100':>12} {'judged@10':>11}"
    print("\n\n=== FINAL RESULTS SUMMARY ===")
    print(header)
    print("-" * len(header))
    for run_name, m in all_metrics.items():
        print(
            f"{run_name:<22} "
            f"{m['nDCG@10']:>10.4f} "
            f"{m['Recall@100']:>12.4f} "
            f"{m['judged@10']:>11.4f}"
        )

    results_path = Path("./results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── 5. Interactive Mode ───────────────────────────────────────────────────
    import sys
    if sys.stdin.isatty() or "google.colab" in sys.modules:
        run_interactive = input("\nLaunch interactive query mode? [y/N]: ").strip().lower()
        if run_interactive == "y":
            interactive_query_mode(bm25_searcher, corpus, top_n=5)
    else:
        print("\nNon-interactive environment detected. Skipping interactive mode.")

if __name__ == "__main__":
    main()