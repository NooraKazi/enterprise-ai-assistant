#!/usr/bin/env python3
"""
Azure AI Search RAG Interface
============================

Production-ready RAG system using Azure AI Search instead of FAISS.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system environment variables

sys.path.append(str(Path(__file__).parent.parent / "llm"))

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from embeddings import EmbeddingGenerator, EmbeddingConfig
from openai_client import LLMClient, ClientConfig

class AzureSearchRAG:
    """RAG system powered by Azure AI Search."""
    
    def __init__(self):
        # Initialize Azure Search client with robust key handling
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        query_key = os.getenv("AZURE_SEARCH_QUERY_KEY")
        admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        index_name = "enterprise-rag-index"
        
        if not endpoint:
            raise ValueError("Missing AZURE_SEARCH_ENDPOINT environment variable")
        
        # Try query key first, then admin key as fallback
        working_key = None
        keys_to_try = []
        if query_key:
            keys_to_try.append(("Query Key", query_key))
        if admin_key:
            keys_to_try.append(("Admin Key", admin_key))
        
        if not keys_to_try:
            raise ValueError("Missing both AZURE_SEARCH_QUERY_KEY and AZURE_SEARCH_ADMIN_KEY environment variables")
        
        for key_type, key_value in keys_to_try:
            try:
                # Test the key works
                test_client = SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(key_value)
                )
                # Try a simple operation
                list(test_client.search("", top=1))
                working_key = key_value
                break
            except Exception:
                continue
        
        if working_key is None:
            raise ValueError("All Azure Search keys failed. Please check your credentials.")
        
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(working_key)
        )
        
        # Initialize embedding and LLM services  
        from embeddings import EmbeddingConfig
        embedding_config = EmbeddingConfig(
            provider="azure", 
            model="text-embedding-3-small",
            min_text_length=1  # Allow short queries like "help"
        )
        self.embedding_service = EmbeddingGenerator(embedding_config)
        
        # Configure LLM client for Azure OpenAI
        llm_config = ClientConfig(
            provider="azure",
            model="gpt-4.1-nano",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        self.llm_client = LLMClient(llm_config)
    
    def search(self, query: str, top_k: int = 3, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """Search using Azure AI Search with vector + text hybrid search."""
        
        # Generate query embedding
        embedding_response = self.embedding_service.generate([query])
        query_vector = embedding_response['data'][0]['embedding']
        
        if use_hybrid:
            # Hybrid search: text + vector  
            from azure.search.documents.models import VectorizedQuery
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["content", "source", "title", "chunkIndex"],
                top=top_k
            )
        else:
            # Pure vector search
            from azure.search.documents.models import VectorizedQuery
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["content", "source", "title", "chunkIndex"],
                top=top_k
            )
        
        return list(results)
    
    def ask(self, question: str, top_k: int = 3) -> str:
        """Ask a question and get RAG-enhanced response with intelligent template routing."""
        
        # Import the sophisticated intent inference system
        sys.path.append(str(Path(__file__).parent.parent / "llm"))
        from prompts import infer_prompt_template, resolve_prompt_template
        
        # Use the intelligent template inference to understand user intent
        inferred_template = infer_prompt_template(question)
        resolved_template = resolve_prompt_template(question)
        
        print(f"🧩 Detected template: {inferred_template.name}")
        print(f"📄 Template file: {Path(inferred_template.file_path).name}")
        print(f"📝 Description: {inferred_template.description}")
        
        # Analyze the question to determine if it's casual conversation or knowledge-seeking
        question_lower = question.lower().strip()
        
        # Detect casual conversation patterns that should use chatbot personality directly
        casual_patterns = [
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "thanks", "thank you", "bye", "goodbye",
            "nice to meet you", "pleased to meet you", "how have you been",
            "good day", "have a nice day", "see you", "take care"
        ]
        
        general_conversation_patterns = [
            "tell me a joke", "how's the weather", "what day is it", "what time",
            "how old are you", "what's your name", "who are you", "who created you",
            "what can you do", "help me with", "i need help", "can you help"
        ]
        
        is_casual_conversation = (
            any(pattern in question_lower for pattern in casual_patterns) or
            any(pattern in question_lower for pattern in general_conversation_patterns) or
            (len(question.split()) <= 5 and any(word in question_lower for word in ["hi", "hello", "hey", "thanks", "bye"]))
        )
        
        # Check for domain-specific knowledge questions that need document search
        domain_keywords = [
            "insurance", "policy", "claim", "coverage", "premium", "deductible", 
            "quote", "underwriting", "liability", "beneficiary", "pol3", "bob", "alice", "john"
        ]
        has_domain_content = any(keyword in question_lower for keyword in domain_keywords)
        
        if is_casual_conversation and not has_domain_content:
            print("💬 Casual conversation detected: Using chatbot personality (no document search)")
            # Use chatbot personality directly for general conversation
            response = self.llm_client.ask(question, prompt_template=inferred_template.name)
            self.last_template_used = inferred_template.name
            self.last_template_file = Path(inferred_template.file_path).name
            return response
        
        # For knowledge questions, use strict document-only mode
        print("🔒 Knowledge question detected: Using document-only mode")
        
        # Route based on intelligent detection
        print("🔍 Searching knowledge base for document-grounded answer...")
            
        # ALWAYS perform vector search for knowledge questions
        search_results = self.search(question, top_k=top_k)
        
        # Display search results for transparency
        print(f"📊 Found {len(search_results)} relevant documents")
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. {result['source']} (score: {result.get('@search.score', 'N/A')})")
        
        # ALWAYS use document context for knowledge questions - never general knowledge
        if search_results:
            # Build context from search results for RAG response  
            context_parts = []
            for i, result in enumerate(search_results, 1):
                source_file = Path(result['source']).name  # Extract just the filename
                context_parts.append(f"Document {i} [Source: {source_file}]:")
                context_parts.append(f"Content: {result['content']}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Generate response using LLM with RAG context - STRICTLY document-based
            prompt = f"""Based ONLY on the following context documents, answer the question: {question}

IMPORTANT: 
1. Only use information from the provided context documents
2. When you reference information, cite the specific document by including the source filename in square brackets [filename]
3. If multiple documents contribute to your answer, cite each relevant source
4. If the context doesn't contain the information needed, say "I don't have enough information in the available documents to answer this question."

Context:
{context}

Answer:"""
            
            response = self.llm_client.ask(prompt, prompt_template="rag")
            
            # Add source summary for transparency
            source_files = [Path(result['source']).name for result in search_results]
            unique_sources = list(dict.fromkeys(source_files))  # Remove duplicates while preserving order
            source_summary = f"\n\n📚 Sources consulted: {', '.join(unique_sources)}"
            response += source_summary
            
            # Always use RAG template for document-grounded answers
            actual_template_used = "rag"
            actual_template_file = "rag_prompt.txt"
            
        else:
            # No relevant documents found for knowledge question
            response = "I don't have any relevant documents in my knowledge base to answer this question."
            response += f"\n\n📚 Sources consulted: No relevant documents found"
            actual_template_used = "rag"
            actual_template_file = "rag_prompt.txt"
        
        # Store template information for display
        self.last_template_used = actual_template_used
        self.last_template_file = actual_template_file
        
        return response

def main():
    """Azure Search RAG with command-line support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure AI Search RAG System")
    parser.add_argument("--query", "-q", type=str, help="Search query to process")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of results to return")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid search (default: True)")
    args = parser.parse_args()
    
    rag = AzureSearchRAG()
    
    if args.query:
        # Command-line mode
        print(f"🔍 Searching Azure AI Search for: '{args.query}'")
        try:
            response = rag.ask(args.query, top_k=args.top_k)
            print(f"🤖 Answer: {response}")
            
            # Show template selection results
            if hasattr(rag, 'last_template_used') and rag.last_template_used:
                print(f"✅ Final template used: {rag.last_template_used} ({rag.last_template_file})")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Interactive mode
    print("🤖 Azure AI Search RAG - Interactive Mode")
    print("💡 Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🤔 Ask anything: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("🔍 Searching Azure AI Search...")
            response = rag.ask(question)
            print(f"🤖 Answer: {response}")
            
            # Show template selection results
            if hasattr(rag, 'last_template_used') and rag.last_template_used:
                print(f"✅ Final template used: {rag.last_template_used} ({rag.last_template_file})")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()