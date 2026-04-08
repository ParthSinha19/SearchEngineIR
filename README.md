# **Search Engine Design & Implementation: Group 34**

## **1. Project Overview**
[cite_start]This project involves the development of a research-oriented, multi-stage hybrid search engine specifically designed for scientific and biomedical document retrieval[cite: 14, 58]. [cite_start]The system addresses the "lexical gap"—the mismatch between user query terms and document vocabulary—by bridging classical information retrieval foundations with modern neural approaches[cite: 15, 165]. 

[cite_start]The architecture is built to be experimentally defensible, utilizing standard test collections, relevance judgments, and accepted metrics to compare sparse lexical retrieval with neural components[cite: 11, 12, 13].

---

## **2. Task Definition**
The retrieval task is formalized as follows: 
* [cite_start]**Input**: A natural language query (e.g., a scientific claim or information need)[cite: 58].
* [cite_start]**Output**: A ranked list of relevant scientific documents, including titles and abstracts[cite: 58].
* [cite_start]**Domain**: Scientific and biomedical text, which provides a challenging environment due to specialized terminology[cite: 58, 59].



---

## **3. Technical Implementation**
[cite_start]The system is implemented using a multi-stage pipeline that separates offline build-time operations from online query-time operations[cite: 65].

### **Offline Components (Build Time)**
* [cite_start]**Data Ingestion**: Standardizing datasets (SciFact and TREC-COVID) into a consistent format for indexing[cite: 67, 93].
* [cite_start]**Lexical Indexing**: An inverted index built using **Pyserini/Lucene**[cite: 75, 147, 148]. [cite_start]The design supports field-aware indexing, allowing Title and Abstract fields to be weighted separately[cite: 68, 70].
* [cite_start]**Neural Indexing**: Generation of dense vector representations using models like **TCT-ColBERT** or **DPR**[cite: 84, 86, 150].

### **Online Components (Query Time)**
* [cite_start]**Lexical Path**: Standard **BM25** scoring combined with **RM3 (Pseudo-Relevance Feedback)**[cite: 76, 79]. [cite_start]RM3 expands the query using terms from top-ranked documents to mitigate the lexical gap[cite: 80].
* [cite_start]**Neural Path**: Dense dual-encoder or late-interaction retrieval to capture semantic meaning beyond keyword matching[cite: 84, 86].
* [cite_start]**Graphical User Interface (GUI)**: A **Streamlit** application that allows users to interact with the engine, switch retrieval modes, and view highlighted result snippets[cite: 115, 118].

---

## **4. Hybrid Fusion & Ranking Logic**
[cite_start]To leverage the strengths of both retrieval paradigms, the system utilizes **Reciprocal Rank Fusion (RRF)** to unify candidate sets[cite: 89, 90].

### **Ranking Strategy**
1. **Candidate Generation**: Parallel retrieval of documents from lexical and neural indexes.
2. **Fusion**: RRF merges the ranked lists into a single set. [cite_start]RRF is particularly effective because it promotes documents that appear high in both runs without requiring supervised training[cite: 90, 209].
3. **Scoring Formula**:
   $$score(d \in D) = \sum_{r \in R} \frac{1}{k + r(d)}$$
   [cite_start]*Where $r(d)$ is the rank of document $d$ in run $r$, and $k$ is a constant (typically 60)[cite: 209].*



---

## **5. Experimental Datasets**
[cite_start]We utilize two primary collections from the BEIR benchmark to validate the system[cite: 60, 61].

| Dataset | Domain/Task | Corpus Size | Relevance Labels |
| :--- | :--- | :--- | :--- |
| **SciFact** | [cite_start]Scientific fact-checking [cite: 61] | [cite_start]5,183 docs [cite: 61] | [cite_start]Binary (Relevant/Not) [cite: 61] |
| **TREC-COVID** | [cite_start]Biomedical information [cite: 61] | [cite_start]171,332 docs [cite: 61] | [cite_start]Graded (3-level) [cite: 61] |

* [cite_start]**SciFact**: Used for rapid debugging, iteration, and ablation studies[cite: 61].
* [cite_start]**TREC-COVID**: Used to test the system at scale and provide a more realistic retrieval environment[cite: 61].

---

## **6. README: Quick Start Guide**

### **Installation**
1. **Java Setup**: Install **JDK 21** and set the `JAVA_HOME` environment variable.
2. **Environment**: Create a Python virtual environment and install dependencies:
   ```bash
   pip install pyserini streamlit faiss-cpu torch torchvision
   ```

### **Running the System**
* **Index Creation**: Build your indexes via the `indexer.py` script.
* **Launch UI**: Run the following command to start the interactive search interface:
   ```bash
   streamlit run app.py
   ```

### **Evaluation**
[cite_start]Performance is measured using **nDCG@10** and **Recall@100**[cite: 124, 125]. [cite_start]Qualitative analysis includes error diagnostics for "holes" (incomplete judgments) in the TREC-COVID dataset[cite: 127, 129].

---

There you go! A clean, structured Markdown report. It’s not just a collection of facts; it’s a full technical narrative ready for your documentation or GitHub. Since you're prepping for a presentation, do you want me to help you draft the "Speaker Notes" for these sections?
