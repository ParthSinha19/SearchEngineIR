import streamlit as st
import time
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from searcher import BM25Searcher, RM3Searcher, DenseSearcher, reciprocal_rank_fusion

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioMed IR System",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 BioMed Information Retrieval System")
st.caption("BM25 · BM25 + RM3 (PRF) · Dense TCT-ColBERT · Hybrid RRF Fusion | SciFact Corpus")

# ── Caching ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    return BM25Searcher(index_path="./indexes/scifact_bm25", top_k=100)

@st.cache_resource(show_spinner="Loading RM3 searcher...")
def load_rm3():
    return RM3Searcher(
        index_path="./indexes/scifact_bm25",
        fb_docs=10,
        fb_terms=10,
        original_query_weight=0.5,
        top_k=100,
    )

@st.cache_resource(show_spinner="Loading Dense model + FAISS index (may take ~30s)...")
def load_dense():
    faiss_index  = "./indexes/scifact_faiss"
    faiss_exists = (
        os.path.exists(os.path.join(faiss_index, "index")) and
        os.path.exists(os.path.join(faiss_index, "docid"))
    )
    if not faiss_exists:
        return None
    return DenseSearcher(
        faiss_index_dir=faiss_index,
        query_encoder_name="castorini/tct_colbert-v2-hnp-msmarco",
        device="cpu",
    )

@st.cache_data(show_spinner="Loading corpus...")
def load_corpus():
    corpus_path = "./data/scifact_pyserini/corpus.jsonl"
    corpus_dict = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus_dict[doc["id"]] = doc
    return corpus_dict

# ── Load everything ───────────────────────────────────────────────────────────
corpus       = load_corpus()
bm25_searcher = load_bm25()
rm3_searcher  = load_rm3()
dense_searcher = load_dense()

dense_available = dense_searcher is not None

# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search Configuration")

    # Build mode options dynamically based on FAISS availability
    mode_options = ["BM25 (Lexical)", "BM25 + RM3 (PRF)"]
    if dense_available:
        mode_options += ["Dense (TCT-ColBERT)", "Hybrid RRF Fusion"]
    else:
        st.warning(
            "FAISS index not found.\n\n"
            "Dense and RRF modes are disabled.\n\n"
            "Run the `pyserini.encode` step to enable them."
        )

    search_mode = st.radio("Retrieval Method", mode_options)

    st.divider()
    st.subheader("Results")
    top_n = st.slider("Documents to display", min_value=1, max_value=20, value=5)

    st.divider()
    st.subheader("RRF Parameters")
    rrf_k = st.slider(
        "RRF k constant",
        min_value=1, max_value=100, value=60,
        help="Higher k reduces the dominance of top-ranked documents. Default: 60 (Cormack et al. 2009)"
    )

    st.divider()
    st.subheader("RM3 Parameters")
    fb_docs  = st.slider("Feedback docs",  min_value=1, max_value=20, value=10,
                         help="Number of top BM25 docs used as pseudo-relevant set")
    fb_terms = st.slider("Expansion terms", min_value=1, max_value=30, value=10,
                         help="Number of new terms injected into the expanded query")
    rm3_alpha = st.slider("Original query weight (α)", min_value=0.0, max_value=1.0,
                          value=0.5, step=0.05,
                          help="1.0 = no expansion, 0.0 = full expansion model")

    # Apply RM3 params live by rebuilding the searcher only when they change
    # We use a session state key to detect changes
    rm3_key = f"rm3_{fb_docs}_{fb_terms}_{rm3_alpha}"
    if st.session_state.get("rm3_key") != rm3_key:
        st.session_state["rm3_key"] = rm3_key
        rm3_searcher = RM3Searcher(
            index_path="./indexes/scifact_bm25",
            fb_docs=fb_docs,
            fb_terms=fb_terms,
            original_query_weight=rm3_alpha,
            top_k=100,
        )

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Hybrid multi-stage IR system evaluated on the "
        "[BEIR SciFact](https://github.com/beir-cellar/beir) benchmark.\n\n"
        "nDCG@10 results:\n"
        "- BM25: **0.6816**\n"
        "- RM3: **0.6576**\n"
        "- Dense: **0.5751**\n"
        "- RRF: **0.6690**"
    )

# ── Main Search UI ────────────────────────────────────────────────────────────
query = st.text_input(
    "🔍 Enter your search query:",
    placeholder="e.g., smoking causes lung cancer | statins lower cholesterol | CRISPR edits human embryos",
)

# Show which mode is active with a coloured badge
mode_colors = {
    "BM25 (Lexical)":       "🟦",
    "BM25 + RM3 (PRF)":     "🟩",
    "Dense (TCT-ColBERT)":  "🟧",
    "Hybrid RRF Fusion":    "🟥",
}
st.markdown(f"{mode_colors.get(search_mode, '⬜')} **Active mode:** {search_mode}")

# ── Execute Search ────────────────────────────────────────────────────────────
if query:
    formatted_query = [("q1", query.lower().strip())]

    with st.spinner(f"Running {search_mode}..."):
        t0 = time.time()

        if search_mode == "BM25 (Lexical)":
            run = bm25_searcher.search(formatted_query)

        elif search_mode == "BM25 + RM3 (PRF)":
            run = rm3_searcher.search(formatted_query)

        elif search_mode == "Dense (TCT-ColBERT)":
            run = dense_searcher.search(formatted_query)

        elif search_mode == "Hybrid RRF Fusion":
            bm25_run  = bm25_searcher.search(formatted_query)
            dense_run = dense_searcher.search(formatted_query)
            run = reciprocal_rank_fusion(bm25_run, dense_run, k=rrf_k)

        results = run.get("q1", [])
        elapsed = time.time() - t0

    # ── Results Header ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Results retrieved", len(results))
    col2.metric("Displayed", min(top_n, len(results)))
    col3.metric("Latency", f"{elapsed:.3f}s")

    st.divider()

    if not results:
        st.warning("No results found. Try a different query.")
    else:
        # ── Compare All Modes button ──────────────────────────────────────────
        if dense_available and st.button("Compare all 4 modes for this query"):
            st.subheader("Side-by-side comparison (Top 3 per mode)")

            bm25_run  = bm25_searcher.search(formatted_query)
            rm3_run   = rm3_searcher.search(formatted_query)
            dense_run = dense_searcher.search(formatted_query)
            rrf_run   = reciprocal_rank_fusion(bm25_run, dense_run, k=rrf_k)

            modes = {
                "BM25":  bm25_run.get("q1", []),
                "RM3":   rm3_run.get("q1", []),
                "Dense": dense_run.get("q1", []),
                "RRF":   rrf_run.get("q1", []),
            }

            cols = st.columns(4)
            for col, (mode_name, mode_results) in zip(cols, modes.items()):
                with col:
                    st.markdown(f"**{mode_name}**")
                    for rank, (docid, score) in enumerate(mode_results[:3], start=1):
                        doc = corpus.get(docid, {})
                        title = doc.get("title", "No Title")[:60] + "..."
                        st.markdown(f"**{rank}.** {title}")
                        st.caption(f"Score: `{score:.4f}`")

            st.divider()

        # ── Individual Results ────────────────────────────────────────────────
        st.subheader(f"Top {min(top_n, len(results))} Results")

        for rank, (docid, score) in enumerate(results[:top_n], start=1):
            doc     = corpus.get(docid, {})
            title   = doc.get("title", "No Title")
            content = doc.get("text") or doc.get("contents", "No abstract available.")

            with st.expander(f"**#{rank} — {title}**", expanded=(rank <= 3)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Abstract**")
                    st.write(content)
                with col_b:
                    st.metric("Rank", f"#{rank}")
                    st.metric("Score", f"{score:.4f}")
                    st.caption(f"Doc ID: `{docid}`")

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.info(
        " **Try searching for:**\n\n"
        "- `smoking causes lung cancer`\n"
        "- `statins lower cholesterol`\n"
        "- `CRISPR can edit human embryos`\n"
        "- `gut microbiome affects mental health`\n"
        "- `sleep deprivation impairs memory`"
    )