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
        """Ask a question and get RAG-enhanced response."""
        
        # 1. Retrieve relevant documents
        search_results = self.search(question, top_k=top_k)
        
        # 2. Build context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Document {i}:")
            context_parts.append(f"Source: {result['source']}")
            context_parts.append(f"Content: {result['content']}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # 3. Generate response using LLM
        prompt = f"""Based on the following context documents, answer the question: {question}

Context:
{context}

Answer:"""
        
        response = self.llm_client.ask(prompt)
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
            
            # Show which prompt template was used
            if hasattr(rag.llm_client.config, 'last_prompt_file') and rag.llm_client.config.last_prompt_file:
                print(f"🧩 Prompt template: {rag.llm_client.config.last_prompt_file}")
                
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
            
            # Show which prompt template was used
            if hasattr(rag.llm_client.config, 'last_prompt_file') and rag.llm_client.config.last_prompt_file:
                print(f"🧩 Prompt template: {rag.llm_client.config.last_prompt_file}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()