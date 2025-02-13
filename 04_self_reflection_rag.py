"""
Self-Reflection RAG (Retrieval-Augmented Generation) Implementation
===================================================================

This module demonstrates a Self-Reflection RAG system, an enhancement over
basic RAG pipelines. In Self-Reflection RAG, the LLM evaluates (or "reflects on")
its own initial answer and may refine it by performing additional retrieval or
revisiting the context. This iterative process helps improve factual accuracy,
reduce hallucinations, and yield more complete responses.

Why Use Self-Reflection RAG?
----------------------------
1. Increased Accuracy: The model can spot and correct mistakes or missing details in its
   first response.
2. Dynamic Context: If the initial retrieval lacks crucial information, the model can
   retrieve additional documents in a second pass.
3. Reduced Hallucinations: By critically evaluating its initial answer, the system can
   detect inconsistencies and refine them.
4. Better User Experience: Users receive more reliable, well-rounded answers without
   needing to re-ask the question multiple times.

Data Source:
------------
We continue to use Lilian Weng's blog post on AI Agents as an example:
  https://lilianweng.github.io/posts/2023-06-23-agent/

Dependencies:
-------------
- langchain, langchain_groq, bs4 (BeautifulSoup4), FAISS, HuggingFace Transformers, python-dotenv

In summary, Self-Reflection RAG iterates over standard RAG steps, adding a reflection
and optional second retrieval + generation pass to improve the final result.
"""

import os
from typing import List, Any, Dict, Tuple
from dataclasses import dataclass
import json

import bs4
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())

@dataclass
class ReflectionMetrics:
    """Stores metrics from the self-reflection process."""
    confidence_score: float
    completeness_score: float
    factual_consistency_score: float
    missing_info_details: List[str]
    improvement_areas: List[str]

class SelfReflectionRAG:
    """
    Enhanced Self-Reflection RAG with confidence scoring and detailed evaluation.
    Implements a multi-step retrieval and refinement process with metrics.
    """

    def __init__(self, retriever: Any, llm: ChatGroq, base_prompt: Any,
                 confidence_threshold: float = 0.8):
        """
        Initialize the enhanced Self-Reflection RAG system.

        Args:
            retriever: Document retriever
            llm: Language model
            base_prompt: Base prompt template
            confidence_threshold: Minimum confidence score to accept answer
        """
        self.retriever = retriever
        self.llm = llm
        self.base_prompt = base_prompt
        self.confidence_threshold = confidence_threshold

    def format_docs(self, docs: List[Any]) -> str:
        """Format documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_initial_response(self, question: str, context: str) -> str:
        """Generate the first-pass response using retrieved context."""
        prompt_input = {
            "context": context,
            "question": question
        }
        prompt_text = self.base_prompt.invoke(prompt_input)
        response = self.llm.invoke(prompt_text)
        return response.content if hasattr(response, 'content') else str(response)

    def evaluate_response(self, question: str, answer: str, context: str) -> ReflectionMetrics:
        """
        Evaluate the response quality using the LLM to generate detailed metrics.
        
        Returns:
            ReflectionMetrics containing scores and improvement suggestions
        """
        evaluation_prompt = f"""
        You are an expert evaluator of AI responses. Analyze this response carefully:

        Question: {question}
        Context provided: {context}
        Answer generated: {answer}

        Evaluate the following aspects and provide a JSON output with these fields:
        1. confidence_score (0-1): How confident are we in this answer?
        2. completeness_score (0-1): Does it fully answer all aspects?
        3. factual_consistency_score (0-1): Is it consistent with the context?
        4. missing_info_details: List specific missing information
        5. improvement_areas: List specific areas for improvement

        JSON output sample:

        Provide your evaluation in valid JSON format with these exact field names. The JSON should be valid and use these *exact* field names.
        """

        # Get evaluation from LLM and Extract content from AIMessage
        evaluation_result = self.llm.invoke(evaluation_prompt)
        evaluation_content = evaluation_result.content if hasattr(evaluation_result, 'content') else str(evaluation_result)
        
        try:
            # Parse the LLM's response as JSON
            metrics_dict = json.loads(evaluation_content)
            
            return ReflectionMetrics(
                confidence_score=float(metrics_dict['confidence_score']),
                completeness_score=float(metrics_dict['completeness_score']),
                factual_consistency_score=float(metrics_dict['factual_consistency_score']),
                missing_info_details=metrics_dict['missing_info_details'],
                improvement_areas=metrics_dict['improvement_areas']
            )
        except json.JSONDecodeError:
            # Fallback values if JSON parsing fails
            return ReflectionMetrics(
                confidence_score=0.5,
                completeness_score=0.5,
                factual_consistency_score=0.5,
                missing_info_details=["Error parsing evaluation metrics"],
                improvement_areas=["Retry evaluation"]
            )

    def reformulate_query(self, original_question: str, metrics: ReflectionMetrics) -> str:
        """
        Reformulate the query based on reflection metrics to get better context.
        """
        reformulation_prompt = f"""
        Original question: {original_question}
        Missing information: {', '.join(metrics.missing_info_details)}
        Improvement areas: {', '.join(metrics.improvement_areas)}

        Please reformulate the question to specifically target the missing information 
        and improvement areas. Provide just the reformulated question without explanation.
        """
        
        reformulated_query = self.llm.invoke(reformulation_prompt)
        return reformulated_query.content if hasattr(reformulated_query, 'content') else str(reformulated_query)

    def retrieve_additional_context(self, reformulated_query: str) -> List[Any]:
        """Retrieve additional context using the reformulated query."""
        return self.retriever.get_relevant_documents(reformulated_query)

    def compare_responses(self, 
                         question: str,
                         first_response: str,
                         second_response: str) -> Tuple[str, float]:
        """
        Compare two responses and select the better one with a confidence score.
        
        Returns:
            Tuple of (better_response, comparison_confidence)
        """
        comparison_prompt = f"""
        Compare these two responses to the question: "{question}"

        First Response:
        {first_response}

        Second Response:
        {second_response}

        Analyze both responses and provide a JSON output with:
        1. better_response: "first" or "second"
        2. confidence: (0-1) how confident are you in this choice?
        3. reasoning: Brief explanation of the choice

        Response in JSON format:
        """

        comparison_result = self.llm.invoke(comparison_prompt)
        comparison_result =  comparison_result.content if hasattr(comparison_result, 'content') else str(comparison_result)
        
        try:
            result = json.loads(comparison_result)
            selected_response = first_response if result['better_response'] == "first" else second_response
            return selected_response, float(result['confidence'])
        except (json.JSONDecodeError, KeyError):
            # Fallback to second response if comparison fails
            return second_response, 0.5

    def run_self_reflection_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Execute the enhanced self-reflection pipeline with detailed metrics.
        
        Returns:
            Dictionary containing final answer and reflection metrics
        """
        # Step 1: Initial retrieval and response
        initial_docs = self.retriever.get_relevant_documents(question)
        initial_context = self.format_docs(initial_docs)
        initial_response = self.generate_initial_response(question, initial_context)

        # Step 2: Evaluate the initial response
        metrics = self.evaluate_response(question, initial_response, initial_context)
        
        # If confidence is high enough, return the initial response
        if metrics.confidence_score >= self.confidence_threshold:
            return {
                "final_answer": initial_response,
                "reflection_metrics": metrics,
                "iterations": 1
            }

        # Step 3: Reformulate and retrieve additional context
        reformulated_query = self.reformulate_query(question, metrics)
        additional_docs = self.retrieve_additional_context(reformulated_query)
        combined_context = self.format_docs(initial_docs + additional_docs)

        # Step 4: Generate refined response
        refined_response = self.generate_initial_response(question, combined_context)
        
        # Step 5: Compare responses and select the better one
        final_answer, comparison_confidence = self.compare_responses(
            question, initial_response, refined_response
        )

        # Final evaluation of the selected answer
        final_metrics = self.evaluate_response(question, final_answer, combined_context)

        return {
            "final_answer": final_answer,
            "reflection_metrics": final_metrics,
            "initial_metrics": metrics,
            "comparison_confidence": comparison_confidence,
            "iterations": 2
        }

def main():
    """Main execution function demonstrating the enhanced Self-Reflection RAG."""
    # Configure environment
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "enhanced-self-reflection-rag"
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['GROQ_API_KEY'] = os.getenv("GROQQ_API_KEY")
    
    # Initialize components
    model_name = "BAAI/bge-small-en"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Load and process documents (example with web content)
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # Create vector store and retriever
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Initialize RAG components
    base_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    # Create Self-Reflection RAG instance
    rag = SelfReflectionRAG(
        retriever=retriever,
        llm=llm,
        base_prompt=base_prompt,
        confidence_threshold=0.8
    )
    
    # Example question
    question = "What are the key components of an AI agent and how do they work together?"
    
    # Run the enhanced pipeline
    result = rag.run_self_reflection_pipeline(question)
    
    # Print results with metrics
    print("\n=== Self-Reflection RAG Results ===")
    print(f"Question: \n{question}")
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print("\nReflection Metrics:")
    print(f"- Confidence Score: {result['reflection_metrics'].confidence_score:.2f}")
    print(f"- Completeness Score: {result['reflection_metrics'].completeness_score:.2f}")
    print(f"- Factual Consistency: {result['reflection_metrics'].factual_consistency_score:.2f}")
    print("\nImprovement Areas:", ", ".join(result['reflection_metrics'].improvement_areas))
    print(f"\nNumber of Iterations: {result['iterations']}")

if __name__ == "__main__":
    main()