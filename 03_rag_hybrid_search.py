"""
Hybrid Search RAG Implementation
==============================

This module implements Hybrid Search in RAG (Retrieval Augmented Generation) combining:
1. Dense Retrieval: Using embeddings (FAISS) for semantic similarity
2. Sparse Retrieval: Using BM25 for keyword matching

When to Use:
-----------
- When dealing with technical documentation where both exact matches (function names, error codes) 
 and conceptual understanding are important
- For complex queries that mix specific terms with broader concepts
- When retrieval accuracy is critical and missing information is costly

Example:
--------
Query: "How to handle JWT token expiration in authentication?"
- Dense Retrieval: Finds conceptually related content about authentication flows and security
- Sparse Retrieval: Finds exact matches for "JWT token expiration" and implementation details
- Combined: Provides both specific implementation guidance and broader security context

Benefits:
--------
- Improved accuracy by combining lexical and semantic matching
- Better handling of technical queries with specific terms
- More robust retrieval across different query types

Implementation Details:
---------------------
- Chunking Strategy: Semantic chunking using RecursiveCharacterTextSplitter
- Chunk Size: 1000 characters with 200 character overlap
- Embedding Model: BAAI/bge-small-en
- Dense Vector Store: FAISS
- Sparse Retrieval: BM25
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
- rank_bm25
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass

import bs4
import numpy as np
from rank_bm25 import BM25Okapi
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
    dense_score: float = 0.0
    sparse_score: float = 0.0
    final_score: float = 0.0

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

class HybridSearchRAG:
    """
    Implements Hybrid Search combining dense and sparse retrievers.
    
    Attributes:
        llm: Language model for answer generation
        vectorstore: FAISS vector store for dense retrieval
        documents: List of document contents for BM25
        alpha: Weight for combining dense and sparse scores
    """
    
    def __init__(self, 
                 llm: ChatGroq,
                 vectorstore: FAISS,
                 documents: List[Any],
                 alpha: float = 0.5):
        self.llm = llm
        self.vectorstore = vectorstore
        self.alpha = alpha
        
        # Prepare documents for BM25
        self.doc_contents = [doc.page_content for doc in documents]
        tokenized_docs = [doc.split() for doc in self.doc_contents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Store original documents for metadata
        self.documents = documents

    def dense_search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Perform dense retrieval using embeddings."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                dense_score=score
            ) for doc, score in results
        ]

    def sparse_search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Perform sparse retrieval using BM25."""
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_k_indices = np.argsort(sparse_scores)[-k:][::-1]
        
        return [
            RetrievedDocument(
                content=self.doc_contents[idx],
                metadata=self.documents[idx].metadata,
                sparse_score=sparse_scores[idx]
            ) for idx in top_k_indices
        ]

    def normalize_scores(self, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Normalize scores to [0, 1] range."""
        if not docs:
            return docs
            
        # Normalize dense scores
        dense_scores = [doc.dense_score for doc in docs]
        min_dense = min(dense_scores)
        max_dense = max(dense_scores)
        dense_range = max_dense - min_dense
        
        # Normalize sparse scores
        sparse_scores = [doc.sparse_score for doc in docs]
        min_sparse = min(sparse_scores)
        max_sparse = max(sparse_scores)
        sparse_range = max_sparse - min_sparse
        
        # Apply normalization
        for doc in docs:
            if dense_range != 0:
                doc.dense_score = (doc.dense_score - min_dense) / dense_range
            if sparse_range != 0:
                doc.sparse_score = (doc.sparse_score - min_sparse) / sparse_range
            
            # Calculate final score using weighted combination
            doc.final_score = (self.alpha * doc.dense_score + 
                             (1 - self.alpha) * doc.sparse_score)
            
        return docs

    def combine_results(self, 
                       dense_docs: List[RetrievedDocument], 
                       sparse_docs: List[RetrievedDocument],
                       k: int = 5) -> List[RetrievedDocument]:
        """Combine and rank results from both retrievers."""
        combined_docs: Dict[str, RetrievedDocument] = {}
        
        # Add dense results
        for doc in dense_docs:
            combined_docs[doc.content] = doc
            
        # Add or update with sparse results
        for doc in sparse_docs:
            if doc.content in combined_docs:
                combined_docs[doc.content].sparse_score = doc.sparse_score
            else:
                combined_docs[doc.content] = doc
        
        # Normalize and calculate final scores
        results = self.normalize_scores(list(combined_docs.values()))
        
        # Sort by final score and return top k
        return sorted(results, 
                     key=lambda x: x.final_score, 
                     reverse=True)[:k]

    def query(self, user_query: str, k: int = 5) -> str:
        """
        Execute hybrid search and generate answer.
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve
            
        Returns:
            Generated answer based on retrieved documents
        """
        # Perform both types of search
        print("\nPerforming dense retrieval...")
        dense_results = self.dense_search(user_query, k)
        
        print("Performing sparse retrieval...")
        sparse_results = self.sparse_search(user_query, k)
        
        # Combine results
        hybrid_results = self.combine_results(dense_results, sparse_results, k)
        
        print("\nRetrieved documents:")
        for i, doc in enumerate(hybrid_results, 1):
            print(f"\nDocument {i}:")
            print(f"Dense Score: {doc.dense_score:.4f}")
            print(f"Sparse Score: {doc.sparse_score:.4f}")
            print(f"Final Score: {doc.final_score:.4f}")
            print(f"Content Preview: {doc.content[:200]}...")
        
        # Generate answer
        context = "\n\n".join(doc.content for doc in hybrid_results)
        
        prompt = f"""Using the following retrieved documents, answer the question.
        Base your answer solely on the provided documents.
        
        Documents:
        {context}
        
        Question: {user_query}
        
        Answer:"""
        
        return self.llm.invoke(prompt).content

def configure_environment():
    """Configure environment variables."""
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_PROJECT'] = 'hybrid-rag'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['GROQ_API_KEY'] = os.getenv("GROQQ_API_KEY")

def main():
    """Main execution function demonstrating Hybrid Search RAG."""
    # Configure environment
    configure_environment()
    
    # Initialize components
    doc_processor = DocumentProcessor(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    text_processor = TextProcessor()
    embedding_manager = EmbeddingManager()
    
    # Load and process documents
    docs = doc_processor.load_documents()
    splits = text_processor.split_documents(docs)
    
    # Create vectorstore
    vectorstore = embedding_manager.create_vectorstore(splits)
    
    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    # Initialize Hybrid Search RAG
    hybrid_rag = HybridSearchRAG(llm, vectorstore, splits)
    
    # Example query
    question = "What are the key components of an AI agent?"
    print(f"\nQuestion: {question}")
    
    # Get answer using hybrid search
    answer = hybrid_rag.query(question)
    print("\nFinal Answer:")
    print(answer)

if __name__ == "__main__":
    main()