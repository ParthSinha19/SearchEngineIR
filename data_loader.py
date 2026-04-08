import logging
from pathlib import Path
from typing import Dict, Tuple, List

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


def download_and_load_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    split: str = "test",
) -> Tuple[Dict, Dict, Dict]:
    dataset_path = Path(data_dir) / dataset_name

    if not dataset_path.exists():
        logger.info(f"Dataset '{dataset_name}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        beir_util.download_and_unzip(url, data_dir)
        logger.info(f"Downloaded to {dataset_path}")
    else:
        logger.info(f"Dataset '{dataset_name}' found at {dataset_path}. Skipping download.")

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)

    logger.info(
        f"Loaded '{dataset_name}' [{split}]: "
        f"{len(corpus)} docs | {len(queries)} queries | {len(qrels)} qrels"
    )

    return corpus, queries, qrels


def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    # Collapse any internal whitespace runs (tabs, newlines) to single spaces
    text = " ".join(text.split())
    return text


def build_pyserini_jsonl_corpus(
    corpus: Dict,
    output_dir: str,
    preprocess: bool = True,
) -> str:
    import json

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = out_path / "corpus.jsonl"

    logger.info(f"Writing Pyserini corpus JSONL to {jsonl_file} ...")

    with open(jsonl_file, "w", encoding="utf-8") as fout:
        for doc_id, doc in corpus.items():
            title = doc.get("title", "") or ""
            text  = doc.get("text", "")  or ""

            if preprocess:
                title = preprocess_text(title)
                text  = preprocess_text(text)

            # `contents` = concatenation used by default BM25 scoring.
            # Keeping title first gives it slight positional emphasis in BM25.
            contents = f"{title} {text}".strip()

            record = {
                "id":       doc_id,
                "contents": contents,
                # Separate fields preserved for potential field-weighted scoring
                "title":    title,
                "text":     text,
            }
            fout.write(json.dumps(record) + "\n")

    logger.info(f"Wrote {len(corpus)} documents to {jsonl_file}")
    return str(out_path)


def get_query_list(queries: Dict, preprocess: bool = True) -> List[Tuple[str, str]]:
    result = []
    for qid, qtext in queries.items():
        if preprocess:
            qtext = preprocess_text(qtext)
        result.append((qid, qtext))
    return sorted(result, key=lambda x: x[0])