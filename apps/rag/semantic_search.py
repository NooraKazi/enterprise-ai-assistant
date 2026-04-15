#!/usr/bin/env python3
"""FAISS-powered semantic search system for the Enterprise AI Assistant.

Implements local vector search using FAISS for similarity search and the existing
embeddings.py for generating embeddings. Supports both in-memory and persistent
vector databases.

Examples:
    python semantic_search.py --build-index sample_docs/ --index-path enterprise.faiss
    python semantic_search.py --query "What is machine learning?" --k 3
    python semantic_search.py --query "AI deployment strategies" --index-path enterprise.faiss
    python semantic_search.py --interactive --provider azure
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import sys

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Import our existing embeddings system
from embeddings import EmbeddingGenerator, EmbeddingConfig, _default_provider, _default_model


@dataclass 
class Document:
    """A document in the semantic search index."""
    
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary."""
        return cls(**data)


@dataclass
class SearchResult:
    """A search result with similarity score."""
    
    document: Document
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'rank': self.rank
        }


class SemanticSearchIndex:
    """FAISS-powered semantic search index."""
    
    def __init__(self, embedding_config: EmbeddingConfig):
        """Initialize the search index with embedding configuration."""
        if faiss is None:
            raise RuntimeError(
                "FAISS not found. Install with: pip install faiss-cpu (or faiss-gpu)"
            )
        if np is None:
            raise RuntimeError(
                "NumPy not found. Install with: pip install numpy"
            )
            
        self.embedding_config = embedding_config
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        self.documents: List[Document] = []
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner Product for cosine similarity
        self.dimension: Optional[int] = None
        
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity using inner product."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        if not documents:
            return
            
        # Generate embeddings for new documents
        texts = [doc.content for doc in documents]
        embedding_result = self.embedding_generator.generate(texts)
        
        # Extract embeddings from the response data structure
        embeddings = embedding_result['data']
        
        # Set dimension if not set
        if self.dimension is None:
            self.dimension = embedding_result['dimensions']
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
        # Add embeddings to documents
        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]['embedding']
            
        # Add to our document store
        self.documents.extend(documents)
        
        # Convert embeddings to numpy array and normalize for cosine similarity
        embedding_matrix = np.array([doc.embedding for doc in documents], dtype=np.float32)
        normalized_embeddings = np.array([self._normalize_vector(emb) for emb in embedding_matrix])
        
        # Add to FAISS index
        self.index.add(normalized_embeddings)
        
        print(f"✅ Added {len(documents)} documents to index (total: {len(self.documents)})")
        
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            print("❌ No documents in index. Add documents first.")
            return []
            
        # Generate embedding for query
        embedding_result = self.embedding_generator.generate([query])
        query_embedding = embedding_result['data'][0]['embedding']
        query_vector = np.array([query_embedding], dtype=np.float32)
        normalized_query = self._normalize_vector(query_vector[0]).reshape(1, -1)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't ask for more results than we have
        scores, indices = self.index.search(normalized_query, k)
        
        # Convert to SearchResult objects
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0:  # Valid result
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank + 1
                ))
                
        return results
    
    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Save FAISS index and document metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Add documents first.")
            
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata (documents and config)
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.meta.json'))
            
        metadata = {
            'documents': [doc.to_dict() for doc in self.documents],
            'embedding_config': asdict(self.embedding_config),
            'dimension': self.dimension
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Saved index to {index_path}")
        print(f"✅ Saved metadata to {metadata_path}")
        
    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Load FAISS index and document metadata from disk."""
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        # Load metadata
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.meta.json'))
            
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Restore documents
        self.documents = [Document.from_dict(doc_data) for doc_data in metadata['documents']]
        self.dimension = metadata['dimension']
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        print(f"✅ Loaded index from {index_path}")
        print(f"📚 Index contains {len(self.documents)} documents")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'embedding_provider': self.embedding_config.provider,
            'embedding_model': self.embedding_config.model
        }


def load_documents_from_directory(directory: Path) -> List[Document]:
    """Load documents from text files in a directory."""
    documents = []
    
    for file_path in directory.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if content:  # Skip empty files
                doc = Document(
                    id=str(file_path.relative_to(directory)),
                    title=file_path.stem,
                    content=content,
                    metadata={
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_type': 'text'
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            
    return documents


def load_documents_from_json(file_path: Path) -> List[Document]:
    """Load documents from a JSON file with array of document objects."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    doc = Document(
                        id=item.get('id', f"doc_{i}"),
                        title=item.get('title', f"Document {i}"),
                        content=item.get('content', str(item)),
                        metadata=item.get('metadata', {})
                    )
                    documents.append(doc)
                else:
                    # Convert non-dict items to documents
                    doc = Document(
                        id=f"doc_{i}",
                        title=f"Document {i}",
                        content=str(item),
                        metadata={}
                    )
                    documents.append(doc)
                    
        return documents
        
    except Exception as e:
        print(f"❌ Failed to load JSON documents from {file_path}: {e}")
        return []


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    sample_docs = [
        {
            "id": "ai_intro",
            "title": "Introduction to Artificial Intelligence",
            "content": "Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. AI systems can be categorized into narrow AI (designed for specific tasks) and general AI (hypothetical systems that could perform any intellectual task).",
            "metadata": {"category": "AI Basics", "difficulty": "beginner"}
        },
        {
            "id": "machine_learning",
            "title": "Machine Learning Fundamentals", 
            "content": "Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions. Common types include supervised learning (labeled data), unsupervised learning (pattern discovery), and reinforcement learning (reward-based learning).",
            "metadata": {"category": "Machine Learning", "difficulty": "intermediate"}
        },
        {
            "id": "deep_learning",
            "title": "Deep Learning and Neural Networks",
            "content": "Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. These networks are inspired by the human brain and can automatically learn hierarchical features from raw data. Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition.",
            "metadata": {"category": "Deep Learning", "difficulty": "advanced"}
        },
        {
            "id": "llm_overview",
            "title": "Large Language Models",
            "content": "Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and generate human-like language. They use transformer architecture and attention mechanisms to process and generate text. Examples include GPT, BERT, and other models that can perform tasks like text completion, question answering, summarization, and code generation.",
            "metadata": {"category": "NLP", "difficulty": "advanced"}
        },
        {
            "id": "ai_deployment",
            "title": "AI Deployment Strategies",
            "content": "Deploying AI models in production requires careful consideration of scalability, performance, monitoring, and maintenance. Common strategies include cloud deployment (AWS, Azure, GCP), edge computing for real-time applications, containerization with Docker and Kubernetes, and API-based services. Monitoring model performance and handling model drift are critical for production systems.",
            "metadata": {"category": "MLOps", "difficulty": "advanced"}
        },
        {
            "id": "vector_search",
            "title": "Vector Search and Embeddings",
            "content": "Vector search, also known as semantic search, uses high-dimensional vector representations (embeddings) to find semantically similar content. Unlike keyword-based search, vector search can understand meaning and context. It's powered by embedding models that convert text, images, or other data into dense vectors that capture semantic meaning. FAISS, Pinecone, and Weaviate are popular vector database solutions.",
            "metadata": {"category": "Search", "difficulty": "intermediate"}
        },
        {
            "id": "rag_systems",
            "title": "Retrieval Augmented Generation",
            "content": "Retrieval Augmented Generation (RAG) combines information retrieval with language generation to provide more accurate and contextual responses. RAG systems first retrieve relevant documents from a knowledge base using vector search, then use this context to generate informed responses. This approach helps reduce hallucinations and provides more factual, up-to-date information than standalone language models.",
            "metadata": {"category": "RAG", "difficulty": "intermediate"}
        }
    ]
    
    return [Document(**doc) for doc in sample_docs]


def print_search_results(results: List[SearchResult], show_content: bool = False) -> None:
    """Print search results in a formatted way."""
    if not results:
        print("❌ No results found.")
        return
        
    print(f"\n🔍 Found {len(results)} results:\n")
    
    for result in results:
        print(f"#{result.rank} 📄 {result.document.title}")
        print(f"   🏷️  ID: {result.document.id}")
        print(f"   📊 Score: {result.score:.4f}")
        
        if result.document.metadata:
            metadata_str = ', '.join([f"{k}: {v}" for k, v in result.document.metadata.items()])
            print(f"   🏷️  Metadata: {metadata_str}")
            
        if show_content:
            content_preview = result.document.content[:200] + "..." if len(result.document.content) > 200 else result.document.content
            print(f"   📖 Content: {content_preview}")
            
        print()


def interactive_mode(search_index: SemanticSearchIndex) -> None:
    """Run interactive search mode."""
    print(f"\n🧠 Semantic Search - Interactive Mode")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'stats' to show index statistics")
    print("💡 Type 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n📋 Available commands:")
                print("  help  - Show this help")
                print("  stats - Show index statistics")
                print("  quit  - Exit interactive mode")
                continue
                
            if query.lower() == 'stats':
                stats = search_index.get_stats()
                print("\n📊 Index Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
                
            if not query:
                continue
                
            # Perform search
            results = search_index.search(query, k=5)
            print_search_results(results, show_content=True)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Search error: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Semantic search with FAISS and OpenAI embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Build index from sample documents
  python semantic_search.py --build-sample --index-path enterprise.faiss
  
  # Build index from directory of text files
  python semantic_search.py --build-index docs/ --index-path enterprise.faiss
  
  # Search existing index
  python semantic_search.py --query "machine learning basics" --index-path enterprise.faiss
  
  # Interactive search mode
  python semantic_search.py --interactive --index-path enterprise.faiss
  
  # Use Azure OpenAI
  python semantic_search.py --provider azure --build-sample --index-path enterprise.faiss
"""
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--build-index', type=Path, metavar='DIR',
                             help='Build index from text files in directory')
    action_group.add_argument('--build-sample', action='store_true',
                             help='Build index from sample AI/ML documents')
    action_group.add_argument('--build-json', type=Path, metavar='FILE',
                             help='Build index from JSON file with documents')
    action_group.add_argument('--query', type=str,
                             help='Search query string')
    action_group.add_argument('--interactive', action='store_true',
                             help='Run interactive search mode')
    action_group.add_argument('--stats', action='store_true',
                             help='Show index statistics')
    
    # Index file arguments
    parser.add_argument('--index-path', type=str, default='semantic_search.faiss',
                       help='Path to FAISS index file (default: semantic_search.faiss)')
    parser.add_argument('--metadata-path', type=str,
                       help='Path to metadata file (default: index_path.meta.json)')
    
    # Search arguments
    parser.add_argument('-k', '--top-k', type=int, default=5,
                       help='Number of results to return (default: 5)')
    parser.add_argument('--show-content', action='store_true',
                       help='Show document content in results')
    
    # Embedding arguments
    parser.add_argument('-p', '--provider', type=str, choices=['openai', 'azure'],
                       help='Embedding provider (default: auto-detect)')
    parser.add_argument('-m', '--model', type=str,
                       help='Embedding model name')
    parser.add_argument('--api-key', type=str,
                       help='API key (or set environment variable)')
    parser.add_argument('--azure-endpoint', type=str,
                       help='Azure OpenAI endpoint')
    
    return parser


def main() -> int:
    """Main function."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        # Build embedding configuration
        provider = args.provider or _default_provider()
        model = args.model or _default_model(provider)
        
        embedding_config = EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=args.api_key,
            azure_endpoint=args.azure_endpoint
        )
        
        # Initialize search index
        search_index = SemanticSearchIndex(embedding_config)
        
        # Handle different actions
        if args.build_index:
            print(f"📂 Loading documents from {args.build_index}")
            documents = load_documents_from_directory(args.build_index)
            if not documents:
                print("❌ No documents found in directory.")
                return 1
            print(f"📚 Found {len(documents)} documents")
            search_index.add_documents(documents)
            search_index.save_index(args.index_path, args.metadata_path)
            
        elif args.build_sample:
            print("📚 Creating sample AI/ML documents")
            documents = create_sample_documents()
            search_index.add_documents(documents)
            search_index.save_index(args.index_path, args.metadata_path)
            
        elif args.build_json:
            print(f"📂 Loading documents from {args.build_json}")
            documents = load_documents_from_json(args.build_json)
            if not documents:
                print("❌ No documents found in JSON file.")
                return 1
            print(f"📚 Found {len(documents)} documents")
            search_index.add_documents(documents)
            search_index.save_index(args.index_path, args.metadata_path)
            
        elif args.query:
            print(f"📖 Loading index from {args.index_path}")
            search_index.load_index(args.index_path, args.metadata_path)
            
            print(f"🔍 Searching for: '{args.query}'")
            results = search_index.search(args.query, args.top_k)
            print_search_results(results, args.show_content)
            
        elif args.interactive:
            print(f"📖 Loading index from {args.index_path}")
            search_index.load_index(args.index_path, args.metadata_path)
            interactive_mode(search_index)
            
        elif args.stats:
            print(f"📖 Loading index from {args.index_path}")
            search_index.load_index(args.index_path, args.metadata_path)
            stats = search_index.get_stats()
            print("\n📊 Index Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())