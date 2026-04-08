import logging
import subprocess
import json
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25Indexer:

    def __init__(
        self,
        input_dir: str,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        title_boost: int = 3,
        threads: int = 4,
        language: str = "en",
    ):
        
        self.input_dir   = Path(input_dir)
        self.index_dir   = Path(index_dir)
        self.k1          = k1
        self.b           = b
        self.title_boost = title_boost
        self.threads     = threads
        self.language    = language

    def _apply_title_boost(self) -> None:
        
        if self.title_boost <= 1:
            return  # No boost needed, contents is already title + text

        src = self.input_dir / "corpus.jsonl"
        tmp = self.input_dir / "corpus_boosted.jsonl"

        logger.info(f"Applying title boost ×{self.title_boost} to corpus JSONL...")

        with open(src, "r") as fin, open(tmp, "w") as fout:
            for line in fin:
                doc = json.loads(line)
                title = doc.get("title", "")
                text  = doc.get("text", "")
                # Title appears title_boost times, then abstract once
                boosted_contents = " ".join([title] * self.title_boost + [text]).strip()
                doc["contents"] = boosted_contents
                fout.write(json.dumps(doc) + "\n")

        # Replace original with boosted version
        tmp.replace(src)
        logger.info("Title boosting complete.")

    def build_index(self) -> str:
        
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._apply_title_boost()

        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection",   "JsonCollection"
            "--collection",   "JsonCollection",
            "--input",        str(self.input_dir),
            "--index",        str(self.index_dir),
            "--generator",    "DefaultLuceneDocumentGenerator",
            "--threads",      str(self.threads),
            "--storePositions",   
            "--storeDocvectors",  
            "--storeRaw",         
        ]

        logger.info(f"Building Lucene index: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Indexing failed:\n{result.stderr}")
            raise RuntimeError("Pyserini indexing subprocess failed.")

        logger.info(f"Index built at: {self.index_dir}")
        logger.info(f"Indexer stdout tail:\n{result.stdout[-500:]}")

        # Persist the BM25 hyperparameters alongside the index for reproducibility
        self._save_config()
        return str(self.index_dir)

    def _save_config(self) -> None:
        
        config = {
            "k1": self.k1,
            "b":  self.b,
            "title_boost": self.title_boost,
        }
        config_path = self.index_dir / "bm25_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved BM25 config to {config_path}")