# Depth and Breadth of RAG Techniques

![image](https://github.com/user-attachments/assets/2d684b68-639a-4398-bf92-8155d6e24157)

## Overview

Welcome to **Depth and Breadth of RAG Techniques**, a collection of hands-on tutorials and code implementations showcasing multiple **Retrieval-Augmented Generation (RAG)** techniques. Each Python script in this repository focuses on a distinct approach to RAG, providing you with a comprehensive view of how to build and extend context-aware question-answering systems.

**Why RAG?**  
RAG is a powerful paradigm that combines large language models (LLMs) with external knowledge sources, enhancing the model's factual accuracy and reducing hallucinations. By integrating retrieval steps into the generation pipeline, your application can dynamically pull in relevant context from large corpora of documents, ensuring more grounded responses.

## Table of Contents
1. [Techniques Covered](#techniques-covered)
2. [Repository Structure](#repository-structure)
3. [Installation & Dependencies](#installation--dependencies)
4. [Usage](#usage)
5. [Detailed Descriptions](#detailed-descriptions-of-each-script)
6. [Future Directions](#future-directions)
7. [License](#license)
8. [Author](#author)

---

## Techniques Covered

1. **Semantic Chunking RAG**  
   - Breaks text into semantically coherent chunks for efficient retrieval.  
2. **Multi-Query & RAG Fusion**  
   - Uses multiple query variations and reciprocal rank fusion for more robust retrieval.  
3. **Hybrid Search RAG**  
   - Combines **dense** (semantic embeddings) and **sparse** (BM25) retrieval methods.  
4. **Self-Reflection RAG**  
   - Enables the LLM to critique and refine its own answer, leading to improved factual correctness.

Each technique is demonstrated in a standalone Python script, making it easier to learn, run, and adapt in your own projects.

---

## Repository Structure

```bash
depth_and_breadth_of_rag_techniques/
├── 01_rag_semantic_chunking.py
├── 02_multiquery_rag_fusion.py
├── 03_rag_hybrid_search.py
├── 04_self_reflection_rag.py
└── README.md
```

- **01_rag_semantic_chunking.py**  
  Basic RAG system featuring semantic chunking for context-aware question answering.

- **02_multiquery_rag_fusion.py**  
  Advanced RAG using multiple queries and RAG fusion (reciprocal rank fusion) for robust retrieval.

- **03_rag_hybrid_search.py**  
  Hybrid RAG approach combining dense (FAISS) and sparse (BM25) retrieval mechanisms.

- **04_self_reflection_rag.py**  
  Self-Reflection RAG that iterates on the initial answer to refine and improve accuracy.

---

## Installation & Dependencies

Before running any of the scripts, make sure you have the required dependencies installed:

- [Python 3.8+](https://www.python.org/downloads/)
- [langchain](https://github.com/hwchase17/langchain)
- [langchain-groq](https://pypi.org/project/langchain-groq/)  
- [BeautifulSoup4 (`bs4`)](https://pypi.org/project/beautifulsoup4/)
- [FAISS](https://github.com/facebookresearch/faiss)  
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [numpy](https://pypi.org/project/numpy/)
- [rank_bm25](https://pypi.org/project/rank-bm25/)

You can install them using:

```bash
pip install langchain langchain-groq bs4 faiss-cpu transformers python-dotenv numpy rank_bm25
```

> **Note**: Depending on your platform, you may need to install FAISS from a specific wheel or via conda (if on Windows).

---

## Usage

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/KaranSharma18/depth_and_breadth_of_rag_techniques.git
   cd depth_and_breadth_of_rag_techniques
   ```

2. **Install Dependencies** (see above).

3. **Run a Specific RAG Technique**  
   For example, to try the **Semantic Chunking RAG**:
   ```bash
   python 01_rag_semantic_chunking.py
   ```
   Adjust paths or environment variables if needed (e.g., `.env` for API keys).

4. **Explore**  
   - Check the console output for the retrieved context and generated answers.
   - Modify parameters (chunk sizes, overlap, embeddings, etc.) within each file to experiment.

---

## Detailed Descriptions of Each Script

### 1. `01_rag_semantic_chunking.py`
- **Purpose**: Demonstrates a basic RAG pipeline with semantic chunking.
- **Key Concepts**:  
  - Chunk Size: 1000 characters  
  - Overlap: 200 characters  
  - Embedding Model: `BAAI/bge-small-en`  
  - Vector Store: FAISS  
  - Foundation for more advanced techniques.

### 2. `02_multiquery_rag_fusion.py`
- **Purpose**: Introduces multi-query expansion and reciprocal rank fusion (RAG Fusion).
- **When to Use**:  
  - Complex questions with multiple aspects.  
  - Robust document ranking needed for ambiguous queries.  
- **Key Concepts**:  
  - Multiple query reformulations to capture diverse contexts.  
  - Reciprocal Rank Fusion for merging different retrieval rankings.

### 3. `03_rag_hybrid_search.py`
- **Purpose**: Combines dense (semantic) and sparse (keyword-based) retrieval methods.
- **Use Case**:  
  - Perfect for technical documentation with specific error codes and conceptual queries.  
  - Merges BM25 and FAISS to improve retrieval accuracy.
- **Key Concepts**:  
  - Dense Vector Store: FAISS  
  - Sparse Retrieval: BM25  
  - Overlapping chunk strategy for maintaining context.

### 4. `04_self_reflection_rag.py`
- **Purpose**: Demonstrates a self-reflection loop where the LLM critiques its own initial answer.
- **Benefits**:  
  - Increased factual accuracy and reduced hallucinations.  
  - Retrieves additional context if the first pass is incomplete.
- **Key Concepts**:  
  - Iterative retrieval and generation cycle.  
  - Self-reflection step to evaluate and refine the answer.

---

## Future Directions

- **Multi-Document Retrieval**: Combine multiple sources to handle cross-referencing.  
- **Re-ranking Mechanisms**: Integrate advanced ranking algorithms like **ColBERT**, **T5**, or **Cross-Encoders**.  
- **Query Expansion & Reformulation**: Automate generation of multiple query variants for more thorough results.  
- **Dynamic Context Window**: Adjust chunk sizes/overlaps dynamically based on the query complexity or length of the source documents.

---

## Author

**[Karan Sharma]**  
- [LinkedIn](https://www.linkedin.com/in/karansharma18/)  
- [GitHub](https://github.com/KaranSharma18)

If you find this repository helpful or have suggestions, please consider giving it a star ★ and opening an issue or pull request. Thanks for visiting!

---
Feel free to reach out for any questions, suggestions, or collaboration opportunities!
