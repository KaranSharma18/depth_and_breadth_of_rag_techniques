"""
Advanced RAG Techniques: Multi-Query and RAG Fusion Implementation
===============================================================

This module implements two advanced RAG (Retrieval Augmented Generation) techniques:
1. Multi-Query RAG: Enhances retrieval through multiple query variations
2. RAG Fusion: Implements Reciprocal Rank Fusion for robust document ranking

When to Use:
-----------
Multi-Query RAG:
- Complex questions with multiple aspects
- When query reformulation could help capture different contexts
- For broader information retrieval needs

RAG Fusion:
- When retrieval robustness is critical
- For ambiguous queries requiring diverse perspectives
- When document ranking quality is paramount

Implementation Details:
---------------------
- Chunking Strategy: Semantic chunking using RecursiveCharacterTextSplitter
- Chunk Size: 1000 characters with 200 character overlap
- Embedding Model: BAAI/bge-small-en
- Vector Store: FAISS
- LLM: Groq's llama3-8b model

Dependencies:
------------
- langchain
- langchain-groq
- bs4 (BeautifulSoup4)
- FAISS
- HuggingFace Transformers
- python-dotenv
- numpy
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass

import bs4
import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(find_dotenv())

@dataclass
class RetrievedDocument:
    """Represents a retrieved document with its content and metadata."""
    content: str
    metadata: Dict[str, Any]
    score: float

class DocumentProcessor:
    """Handles document loading and preprocessing."""
    
    def __init__(self, url: str):
        self.url = url
        self.bs_kwargs = dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    
    def load_documents(self) -> List[Any]:
        loader = WebBaseLoader(
            web_paths=(self.url,),
            bs_kwargs=self.bs_kwargs
        )
        return loader.load()

class TextProcessor:
    """Handles text splitting operations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def split_documents(self, docs: List[Any]) -> List[Any]:
        return self.text_splitter.split_documents(docs)

class EmbeddingManager:
    """Manages document embedding and vector storage."""
    
    def __init__(self):
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
    
    def create_vectorstore(self, documents: List[Any]) -> FAISS:
        return FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

class MultiQueryRAG:
    """
    Implements Multi-Query RAG technique for enhanced document retrieval.
    Generates multiple variations of the original query to improve retrieval coverage.
    """
    
    def __init__(self, llm: ChatGroq, vectorstore: FAISS, num_queries: int = 5):
        self.llm = llm
        self.vectorstore = vectorstore
        self.num_queries = num_queries

    def generate_queries(self, original_query: str) -> List[str]:
        """Generate multiple variations of the original query."""
        prompt = f"""Generate {self.num_queries} different versions of the following query
        that capture the same meaning but with different wording.
        Each version should:
        - Maintain the core intent of the original query
        - Use different terminology or phrasing
        - Be self-contained and grammatically correct

        Do not write something like Here are 5 different versions of the query. Only provide the queries.
        
        Original query: {original_query}
        
        Provide these alternative questions separated by newlines."""
        
        response = self.llm.invoke(prompt).content
        return [q.strip() for q in response.split('\n') if q.strip()]

    def retrieve(self, query: str, num_docs: int = 3) -> List[RetrievedDocument]:
        """Retrieve documents for a single query."""
        results = self.vectorstore.similarity_search_with_score(query, k=num_docs)
        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ) for doc, score in results
        ]

    def query(self, user_query: str, num_docs: int = 3) -> List[RetrievedDocument]:
        """Perform multi-query retrieval and result aggregation."""
        queries = self.generate_queries(user_query)
        all_docs = []
        
        print(f"Generated queries:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
            
        for query in queries:
            docs = self.retrieve(query, num_docs)
            all_docs.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.content not in seen:
                seen.add(doc.content)
                unique_docs.append(doc)
        
        return unique_docs[:num_docs]

class RAGFusion:
    """
    Implements RAG Fusion using Reciprocal Rank Fusion (RRF) scoring.
    Combines results from multiple queries for more robust document retrieval.
    """
    
    def __init__(self, llm: ChatGroq, vectorstore: FAISS, k: int = 60, num_queries: int = 5):
        self.llm = llm
        self.vectorstore = vectorstore
        self.k = k  # RRF constant
        self.num_queries = num_queries
    
    def get_answer_from_llm(self, llm: ChatGroq, question: str, documents: List[RetrievedDocument]) -> str:
        """Generate an answer using the LLM based on retrieved documents."""
        # Combine document contents
        context = "\n\n".join(doc.content for doc in documents)
        
        prompt = f"""Using the following documents, answer the question. Base your answer solely on the provided documents.
        If the documents don't contain relevant information, say so.

        Documents:
        {context}

        Question: {question}

        Answer: """
        
        return llm.invoke(prompt).content

    def calculate_rrf_score(self, ranks: List[int]) -> float:
        """Calculate RRF score for a document based on its ranks."""
        return sum(1 / (rank + self.k) for rank in ranks)

    def query(self, user_query: str, num_docs: int = 3) -> List[RetrievedDocument]:
        """Perform RAG Fusion retrieval with RRF scoring."""
        # Generate multiple queries
        mq_rag = MultiQueryRAG(self.llm, self.vectorstore, self.num_queries)
        queries = mq_rag.generate_queries(user_query)
        
        print(f"Generated queries:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
        
        # Track document rankings
        doc_ranks: Dict[str, List[int]] = {}
        
        # Retrieve and track rankings
        for query in queries:
            results = self.vectorstore.similarity_search_with_score(query, k=num_docs)
            for rank, (doc, score) in enumerate(results):
                if doc.page_content not in doc_ranks:
                    doc_ranks[doc.page_content] = []
                doc_ranks[doc.page_content].append(rank)
        
        # Calculate RRF scores
        doc_scores = {
            content: self.calculate_rrf_score(ranks)
            for content, ranks in doc_ranks.items()
        }
        
        # Sort by RRF scores
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            RetrievedDocument(
                content=content,
                metadata={},
                score=score
            )
            for content, score in sorted_docs[:num_docs]
        ]

def main():
    """Main execution function demonstrating Multi-Query RAG and RAG Fusion."""
    # Configure environment
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_PROJECT'] = 'advanced-rag'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['GROQ_API_KEY'] = os.getenv("GROQQ_API_KEY")
    
    # Initialize components
    doc_processor = DocumentProcessor(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    text_processor = TextProcessor()
    embedding_manager = EmbeddingManager()
    
    # Load and process documents
    docs = doc_processor.load_documents()
    splits = text_processor.split_documents(docs)
    vectorstore = embedding_manager.create_vectorstore(splits)
    
    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    # Example query
    question = "What are the key components of an AI agent?"
    
    print("\n=== Multi-Query RAG ===")
    multi_query_rag = MultiQueryRAG(llm, vectorstore)
    mq_docs = multi_query_rag.query(question)
    print("\nRetrieved documents:")
    for i, doc in enumerate(mq_docs, 1):
        print(f"\nDocument {i} (score: {doc.score:.4f}):")
        print(doc.content[:200] + "...")
    
    print("\n=== RAG Fusion ===")
    rag_fusion = RAGFusion(llm, vectorstore)
    fusion_docs = rag_fusion.query(question)
    print("\nRetrieved documents:")
    for i, doc in enumerate(fusion_docs, 1):
        print(f"\nDocument {i} (score: {doc.score:.4f}):")
        print(doc.content[:200] + "...")

    # Generate answer using RAG Fusion results
    fusion_answer = rag_fusion.get_answer_from_llm(llm, question, fusion_docs)
    print(f"Question: {question}\nAnswer: {fusion_answer}")

if __name__ == "__main__":
    main()