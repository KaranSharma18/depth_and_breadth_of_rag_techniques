"""
RAG (Retrieval Augmented Generation) Implementation
==========================================================

This module implements a basic RAG (Retrieval Augmented Generation) system using semantic chunking
for enhanced context-aware question answering.

This implementation serves as a foundation for more advanced RAG techniques and can be extended
with features like:
- Multi-document retrieval
- Re-ranking mechanisms
- Hybrid search (combining dense and sparse retrievers)
- Dynamic context window adjustment
- Query expansion and reformulation

The system demonstrates basic RAG capabilities and particularly useful for processing long-form content and
providing contextually relevant answers based on the source material.

Data Source:
-----------
The system processes content from Lilian Weng's blog post on AI Agents:
https://lilianweng.github.io/posts/2023-06-23-agent/

RAG Implementation Details:
-------------------------
- Chunking Strategy: Semantic chunking using RecursiveCharacterTextSplitter
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters (20% overlap to maintain context between chunks)
- Embedding Model: BAAI/bge-small-en (Efficient embedding model)
- Vector Store: FAISS (In-memory vector storage for efficient similarity search)

Main Pipeline Steps:
------------------
1. Document loading and preprocessing from the blog
2. Text chunking with overlap for context preservation
3. Vector embedding and storage
4. Context-aware retrieval and question answering

Dependencies:
- langchain
- langchain-groq
- bs4 (BeautifulSoup4)
- FAISS
- HuggingFace Transformers
- python-dotenv
"""

import os
from typing import List, Dict, Any

import bs4
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Configure LangChain settings for monitoring and API access
def configure_environment() -> None:
    """
    Configure environment variables for LangChain and Groq.
    Enables tracing and sets up API endpoints.
    """
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_PROJECT'] = 'basic-rag'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['GROQ_API_KEY'] = os.getenv("GROQQ_API_KEY")

class DocumentProcessor:
    """Handles document loading and preprocessing for the RAG pipeline."""
    
    def __init__(self, url: str):
        """
        Initialize the document processor.
        
        Args:
            url: Web URL to load documents from
        """
        self.url = url
        # Configure BeautifulSoup to parse specific HTML classes
        self.bs_kwargs = dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    
    def load_documents(self) -> List[Any]:
        """
        Load and preprocess documents from the specified URL.
        
        Returns:
            List of loaded documents
        """
        loader = WebBaseLoader(
            web_paths=(self.url,),
            bs_kwargs=self.bs_kwargs
        )
        return loader.load()

class TextProcessor:
    """Handles text splitting and embedding operations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text processor with chunking parameters.
        
        Args:
            chunk_size: Size of text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def split_documents(self, docs: List[Any]) -> List[Any]:
        """
        Split documents into overlapping chunks.
        
        Args:
            docs: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(docs)

class EmbeddingManager:
    """Manages document embedding and vector storage."""
    
    def __init__(self):
        """Initialize embedding model with BGE-small-en."""
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
    
    def create_vectorstore(self, documents: List[Any]) -> FAISS:
        """
        Create FAISS vectorstore from documents.
        
        Args:
            documents: List of document chunks to embed
            
        Returns:
            FAISS vectorstore instance
        """
        return FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

class RAGPipeline:
    """Main RAG pipeline implementation."""
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        # Load RAG prompt template from LangChain hub
        self.prompt = hub.pull("rlm/rag-prompt")
        # Initialize Groq LLM with llama3-8b model
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    @staticmethod
    def format_docs(docs: List[Any]) -> str:
        """
        Format retrieved documents into a single string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted string of document contents
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self, retriever: Any) -> Any:
        """
        Create the RAG chain combining retrieval and generation.
        
        Args:
            retriever: Document retriever instance
            
        Returns:
            Composed RAG chain
        """
        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

def main():
    """Main execution function."""
    # Configure environment
    configure_environment()
    
    # Initialize pipeline components
    doc_processor = DocumentProcessor(
        "https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    text_processor = TextProcessor()
    embedding_manager = EmbeddingManager()
    rag_pipeline = RAGPipeline()
    
    # Load and process documents
    docs = doc_processor.load_documents()
    splits = text_processor.split_documents(docs)
    
    # Create vectorstore and retriever
    vectorstore = embedding_manager.create_vectorstore(splits)
    retriever = vectorstore.as_retriever()
    
    # Create and execute RAG chain
    rag_chain = rag_pipeline.create_chain(retriever)
    
    # Example query
    question = "What is an AI agent?"
    answer = rag_chain.invoke(question)
    print(f"Question: {question}\nAnswer: {answer}")

if __name__ == "__main__":
    main()