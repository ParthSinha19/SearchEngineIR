import streamlit as st
import time
from searcher import BM25Searcher, DenseSearcher

# 1. Page Config
st.set_page_config(page_title="BioMed Search", page_icon="🔍", layout="centered")
st.title("BioMed Search Engine")
st.caption("Powered by BM25 and TCT-ColBERT Dense Retrieval")

# 2. CACHING (Crucial!)
# We use @st.cache_resource so the FAISS index and PyTorch model 
# only load into memory ONCE, not every time the user types a query.
@st.cache_resource
def load_models():
    bm25 = BM25Searcher(index_path="./indexes/scifact_bm25", top_k=10)
    dense = DenseSearcher(
        faiss_index_dir="./indexes/scifact_faiss", 
        query_encoder_name="castorini/tct_colbert-v2-hnp-msmarco",
        device="cuda" # or cpu
    )
    # You would also load your corpus dictionary here to get the actual text
    # corpus = load_my_corpus() 
    return bm25, dense

bm25_searcher, dense_searcher = load_models()

# 3. The UI
search_mode = st.radio("Select Retrieval Method:", ["BM25 (Lexical)", "Dense (Semantic)"], horizontal=True)
query = st.text_input("Enter your search query:", placeholder="e.g., What is the impact of hypertension on COVID-19?")

# 4. Execute Search
if query:
    with st.spinner("Searching millions of documents..."):
        t0 = time.time()
        
        # Format the query the way your searcher expects it: [(qid, qtext)]
        formatted_query = [("q1", query)]
        
        if search_mode == "BM25 (Lexical)":
            run = bm25_searcher.search(formatted_query)
        else:
            run = dense_searcher.search(formatted_query)
            
        results = run.get("q1", [])
        elapsed = time.time() - t0
        
    # 5. Display Results
    st.success(f"Retrieved {len(results)} results in {elapsed:.3f} seconds.")
    
    for rank, (docid, score) in enumerate(results, start=1):
        with st.container():
            st.markdown(f"### {rank}. Document ID: `{docid}`")
            st.write(f"**Score:** {score:.4f}")
            # If you loaded the corpus dict, you'd print the title/abstract here:
            # st.write(corpus[docid]['text']) 
            st.divider()