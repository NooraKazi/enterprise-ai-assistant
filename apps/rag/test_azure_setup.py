#!/usr/bin/env python3
"""
Azure OpenAI Configuration Test
===============================

Test your Azure OpenAI setup with text-embedding-3-small model.
Verifies API keys and embedding generation before running the full RAG system.

Prerequisites:
- Azure OpenAI resource with text-embedding-3-small deployment
- Environment variables set:
  * AZURE_OPENAI_API_KEY
  * AZURE_OPENAI_ENDPOINT

Usage:
    python test_azure_setup.py
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system environment variables

# Add to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from embeddings import EmbeddingGenerator, EmbeddingConfig
    from mini_rag import MiniRAG, RAGConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def check_azure_environment():
    """Check if Azure OpenAI environment variables are set."""
    print("🔍 Checking Azure OpenAI Environment Variables")
    print("=" * 50)
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    if api_key:
        print("✅ AZURE_OPENAI_API_KEY found")
    else:
        print("❌ AZURE_OPENAI_API_KEY not set!")
        return False
    
    if endpoint:
        print(f"✅ AZURE_OPENAI_ENDPOINT found: {endpoint}")
    else:
        print("❌ AZURE_OPENAI_ENDPOINT not set!")
        return False
    
    if deployment:
        print(f"✅ AZURE_OPENAI_EMBEDDING_DEPLOYMENT found: {deployment}")
    else:
        print("⚠️ AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set (will use model name)")
    
    return True


def test_azure_embeddings():
    """Test Azure OpenAI embedding generation."""
    print("\n🧪 Testing Azure OpenAI Embeddings")
    print("=" * 40)
    
    try:
        # Configure for Azure OpenAI
        config = EmbeddingConfig(
            provider="azure",
            model="text-embedding-3-small"  # Your deployment name
        )
        
        generator = EmbeddingGenerator(config)
        
        # Test with sample insurance-related text
        test_texts = [
            "auto insurance policy coverage",
            "home insurance benefits",
            "life insurance premium"
        ]
        
        print(f"📝 Generating embeddings for {len(test_texts)} test texts...")
        
        result = generator.generate(test_texts)
        
        print(f"✅ Embeddings generated successfully!")
        print(f"   Model: {result['model']}")
        print(f"   Provider: {result['provider']}")
        print(f"   Dimensions: {result['dimensions']}")
        print(f"   Texts processed: {len(result['data'])}")
        
        # Show usage info
        if 'usage' in result:
            print(f"   Tokens used: {result['usage']['total_tokens']}")
            cost_estimate = result['usage']['total_tokens'] * 0.0001 / 1000  # Rough estimate
            print(f"   Estimated cost: ${cost_estimate:.6f}")
        
        # Show sample embedding
        if result['data']:
            sample_embedding = result['data'][0]['embedding'][:5]  # First 5 dimensions
            print(f"   Sample embedding: [{', '.join(f'{x:.4f}' for x in sample_embedding)}...]")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False


def test_rag_initialization():
    """Test RAG system initialization with Azure OpenAI."""
    print("\n🤖 Testing RAG System Initialization")
    print("=" * 40)
    
    try:
        # Configure RAG for Azure OpenAI
        config = RAGConfig(
            provider="azure",
            index_path="test_azure_rag_index.faiss"
        )
        
        rag = MiniRAG(config)
        
        print("✅ RAG system initialized successfully with Azure OpenAI!")
        
        if rag.search_system:
            print("✅ Search system ready")
        else:
            print("ℹ️ Search system not ready (no existing index)")
        
        if rag.llm_client:
            print("✅ LLM client initialized")
        else:
            print("⚠️ LLM client not initialized (check Azure OpenAI LLM configuration)")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        return False


def show_next_steps():
    """Show next steps for using the RAG system."""
    print("\n🎯 Next Steps")
    print("=" * 15)
    print("1. Build knowledge base from your insurance data:")
    print("   python mini_rag.py --build-index --data ../../data/insurance_data.json")
    print()
    print("2. Try interactive questioning:")
    print("   python mini_rag.py --interactive")
    print()
    print("3. Ask single questions:")
    print('   python mini_rag.py --query "What auto insurance policies do you have?"')
    print()
    print("📋 Your Azure Configuration:")
    print(f"   Provider: Azure OpenAI")
    print(f"   Embedding Model: text-embedding-3-small")
    print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')}")


def main():
    """Run all tests."""
    print("🚀 Azure OpenAI RAG System Test")
    print("=" * 35)
    
    # Test environment variables
    if not check_azure_environment():
        print("\n❌ Environment setup incomplete!")
        print("Please set the required Azure OpenAI environment variables:")
        print("  AZURE_OPENAI_API_KEY=your-api-key")
        print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        return 1
    
    # Test embeddings
    if not test_azure_embeddings():
        print("\n❌ Embedding test failed!")
        print("Check your Azure OpenAI configuration and try again.")
        return 1
    
    # Test RAG initialization  
    if not test_rag_initialization():
        print("\n❌ RAG initialization failed!")
        return 1
    
    print("\n🎉 All tests passed! Your Azure OpenAI setup is working correctly.")
    
    # Show next steps
    show_next_steps()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())