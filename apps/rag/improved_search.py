#!/usr/bin/env python3
"""
Enhanced Semantic Search System with Improved Ranking
=====================================================

Advanced semantic search with better ranking, chunking integration, and 
multiple scoring mechanisms for improved RAG results.

Features:
- Multiple similarity metrics (cosine, dot product, euclidean)
- Hybrid search combining semantic + keyword ranking
- Re-ranking with multiple factors
- Chunking integration for better document processing
- Query expansion and refinement
- Advanced result filtering and boosting

Examples:
    # Build index with chunking
    python improved_search.py --build-with-chunks --input documents/ --index-path improved.faiss
    
    # Search with re-ranking
    python improved_search.py --query "machine learning" --rerank --boost-recent --index-path improved.faiss
    
    # Hybrid search (semantic + keyword)
    python improved_search.py --query "AI deployment" --hybrid --alpha 0.7 --index-path improved.faiss
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter
import os
import sys
import re
from datetime import datetime

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
except ImportError:
    sklearn = None
    TfidfVectorizer = None
    sklearn_cosine_similarity = None

# Import our existing systems
from embeddings import EmbeddingGenerator, EmbeddingConfig, _default_provider, _default_model
from chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy, Chunk


@dataclass
class EnhancedDocument:
    """Enhanced document with chunking and metadata information."""
    
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    # Enhanced fields for better ranking
    chunk_info: Optional[Dict[str, Any]] = None  # Information about source chunk
    parent_document: Optional[str] = None        # Original document if this is a chunk
    keywords: List[str] = None                   # Extracted keywords
    created_date: Optional[datetime] = None      # Document creation date
    boost_factor: float = 1.0                   # Manual boost for ranking
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.chunk_info is None:
            self.chunk_info = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Handle datetime serialization
        if self.created_date:
            data['created_date'] = self.created_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedDocument':
        """Create EnhancedDocument from dictionary."""
        # Handle datetime deserialization 
        if 'created_date' in data and data['created_date']:
            if isinstance(data['created_date'], str):
                data['created_date'] = datetime.fromisoformat(data['created_date'])
        return cls(**data)


@dataclass  
class EnhancedSearchResult:
    """Enhanced search result with multiple scoring components."""
    
    document: EnhancedDocument
    semantic_score: float           # Original cosine similarity
    keyword_score: float = 0.0      # TF-IDF based keyword score
    boost_score: float = 1.0        # Document boost factor
    recency_score: float = 1.0      # Recency-based score
    final_score: float = 0.0        # Combined final score
    rank: int = 0
    explanation: Dict[str, Any] = None  # Scoring explanation
    
    def __post_init__(self):
        if self.explanation is None:
            self.explanation = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document': self.document.to_dict(),
            'semantic_score': self.semantic_score,
            'keyword_score': self.keyword_score,
            'boost_score': self.boost_score,
            'recency_score': self.recency_score,
            'final_score': self.final_score,
            'rank': self.rank,
            'explanation': self.explanation
        }


@dataclass
class SearchConfig:
    """Configuration for enhanced search."""
    
    # Similarity metrics
    similarity_metric: str = "cosine"  # cosine, dot_product, euclidean
    
    # Hybrid search parameters
    enable_hybrid: bool = False
    semantic_weight: float = 0.7       # Weight for semantic similarity
    keyword_weight: float = 0.3        # Weight for keyword similarity
    
    # Re-ranking parameters
    enable_reranking: bool = True
    boost_recent: bool = False         # Boost recently created documents
    recency_decay_days: int = 365      # Days for recency decay
    
    # Query expansion
    enable_query_expansion: bool = False
    expansion_terms: int = 3           # Number of terms to add
    
    # Result filtering
    min_score_threshold: float = 0.1   # Minimum similarity threshold
    max_results: int = 20              # Maximum results to consider
    diversity_threshold: float = 0.9   # Filter very similar results


class EnhancedSemanticSearch:
    """Enhanced semantic search with better ranking and chunking integration."""
    
    def __init__(self, embedding_config: EmbeddingConfig, search_config: SearchConfig = None):
        """Initialize enhanced search system."""
        if faiss is None:
            raise RuntimeError("FAISS not found. Install with: pip install faiss-cpu")
        if np is None:
            raise RuntimeError("NumPy not found. Install with: pip install numpy")
        
        self.embedding_config = embedding_config
        self.search_config = search_config or SearchConfig()
        self.embedding_generator = EmbeddingGenerator(embedding_config)
        
        # Search components
        self.documents: List[EnhancedDocument] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: Optional[int] = None
        
        # TF-IDF for keyword search
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # Chunking integration
        self.chunker: Optional[DocumentChunker] = None
    
    def set_chunking_config(self, chunking_config: ChunkingConfig) -> None:
        """Set chunking configuration for document processing."""
        self.chunker = DocumentChunker(chunking_config)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def _extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract key terms from text using simple frequency analysis."""
        # Simple keyword extraction (could be enhanced with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'this', 'that', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'did', 'does', 'do', 'get', 'got', 'make',
                     'made', 'take', 'took', 'see', 'saw', 'know', 'knew', 'think', 'thought', 'say', 'said'}
        
        words = [w for w in words if w not in stop_words and len(w) > 3]
        word_counts = Counter(words)
        
        return [word for word, count in word_counts.most_common(num_keywords)]
    
    def _build_tfidf_index(self) -> None:
        """Build TF-IDF index for keyword search."""
        if not TfidfVectorizer:
            print("⚠️  scikit-learn not available. Keyword search disabled.")
            return
        
        if not self.documents:
            return
        
        # Prepare documents for TF-IDF
        texts = [doc.content for doc in self.documents]
        
        # Build TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english', 
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print(f"✅ Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
    
    def add_documents(self, documents: List[EnhancedDocument], use_chunking: bool = False) -> None:
        """Add documents to the index with optional chunking."""
        if not documents:
            return
        
        final_documents = []
        
        if use_chunking and self.chunker:
            # Process documents with chunking
            print(f"🔪 Chunking {len(documents)} documents...")
            
            for doc in documents:
                chunks = self.chunker.chunk_document(doc.content, doc.id)
                
                for chunk in chunks:
                    # Create enhanced document from chunk
                    chunk_doc = EnhancedDocument(
                        id=f"{doc.id}_chunk_{chunk.chunk_index}",
                        title=f"{doc.title} (Part {chunk.chunk_index + 1})",
                        content=chunk.text,
                        metadata=doc.metadata.copy(),
                        parent_document=doc.id,
                        chunk_info={
                            "chunk_index": chunk.chunk_index,
                            "char_count": chunk.char_count,
                            "token_count": chunk.token_count,
                            "overlap": chunk.overlap_with_previous,
                            "strategy": chunk.metadata.get("strategy", "unknown")
                        },
                        keywords=self._extract_keywords(chunk.text),
                        created_date=doc.created_date,
                        boost_factor=doc.boost_factor
                    )
                    final_documents.append(chunk_doc)
            
            print(f"📚 Created {len(final_documents)} chunks from {len(documents)} documents")
        else:
            # Use documents as-is, but extract keywords
            for doc in documents:
                if not doc.keywords:
                    doc.keywords = self._extract_keywords(doc.content)
                final_documents.append(doc)
        
        # Generate embeddings
        texts = [doc.content for doc in final_documents]
        embedding_result = self.embedding_generator.generate(texts)
        embeddings = embedding_result['data']
        
        # Set dimension if not set
        if self.dimension is None:
            self.dimension = embedding_result['dimensions']
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to documents
        for i, doc in enumerate(final_documents):
            doc.embedding = embeddings[i]['embedding']
        
        # Add to document store
        self.documents.extend(final_documents)
        
        # Create embedding matrix and add to FAISS
        embedding_matrix = np.array([doc.embedding for doc in final_documents], dtype=np.float32)
        normalized_embeddings = np.array([self._normalize_vector(emb) for emb in embedding_matrix])
        self.index.add(normalized_embeddings)
        
        # Rebuild TF-IDF index
        if self.search_config.enable_hybrid:
            self._build_tfidf_index()
        
        print(f"✅ Added {len(final_documents)} documents to index (total: {len(self.documents)})")
    
    def _compute_similarity_scores(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute similarity scores using configured metric."""
        query_vector = query_embedding.reshape(1, -1)
        
        if self.search_config.similarity_metric == "cosine":
            # Use normalized vectors with inner product for cosine similarity
            normalized_query = self._normalize_vector(query_embedding).reshape(1, -1)
            scores, indices = self.index.search(normalized_query, k)
        
        elif self.search_config.similarity_metric == "dot_product":
            # Raw dot product (no normalization)
            scores, indices = self.index.search(query_vector, k)
            
        elif self.search_config.similarity_metric == "euclidean":
            # Convert to euclidean distance (lower is better, so negate)
            # This requires a different FAISS index type, so fall back to cosine
            print("⚠️  Euclidean distance not implemented, using cosine similarity")
            normalized_query = self._normalize_vector(query_embedding).reshape(1, -1)
            scores, indices = self.index.search(normalized_query, k)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.search_config.similarity_metric}")
        
        return scores, indices
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Perform TF-IDF based keyword search."""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Compute similarities with all documents
        similarities = sklearn_cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0]
        
        return results
    
    def _compute_recency_score(self, document: EnhancedDocument) -> float:
        """Compute recency-based score boost."""
        if not document.created_date or not self.search_config.boost_recent:
            return 1.0
        
        days_old = (datetime.now() - document.created_date).days
        decay_factor = math.exp(-days_old / self.search_config.recency_decay_days)
        
        # Scale between 1.0 and 2.0
        return 1.0 + decay_factor
    
    def _rerank_results(self, results: List[EnhancedSearchResult], query: str) -> List[EnhancedSearchResult]:
        """Re-rank results using multiple factors."""
        for result in results:
            # If final score isn't set yet, calculate it
            if result.final_score == 0.0:
                # Compute recency score
                result.recency_score = self._compute_recency_score(result.document)
                
                # Apply document boost
                result.boost_score = result.document.boost_factor
                
                # Combine scores
                if self.search_config.enable_hybrid:
                    # Weighted combination of semantic and keyword scores
                    combined_similarity = (
                        self.search_config.semantic_weight * result.semantic_score +
                        self.search_config.keyword_weight * result.keyword_score
                    )
                else:
                    combined_similarity = result.semantic_score
                
                # Apply boosts only if re-ranking is enabled
                if self.search_config.enable_reranking:
                    result.final_score = combined_similarity * result.recency_score * result.boost_score
                else:
                    result.final_score = combined_similarity
                
                # Store explanation
                result.explanation = {
                    "base_similarity": combined_similarity,
                    "recency_boost": result.recency_score,
                    "manual_boost": result.boost_score,
                    "reranking_enabled": self.search_config.enable_reranking,
                    "hybrid_weights": {
                        "semantic": self.search_config.semantic_weight,
                        "keyword": self.search_config.keyword_weight
                    } if self.search_config.enable_hybrid else None
                }
        
        # Sort by final score
        results.sort(key=lambda r: r.final_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _filter_diverse_results(self, results: List[EnhancedSearchResult]) -> List[EnhancedSearchResult]:
        """Filter out very similar results to promote diversity."""
        if len(results) <= 1:
            return results
        
        filtered = [results[0]]  # Always keep top result
        
        for candidate in results[1:]:
            # Check similarity to already selected results
            too_similar = False
            
            for selected in filtered:
                # Simple similarity check based on shared keywords or content overlap
                candidate_keywords = set(candidate.document.keywords[:5])
                selected_keywords = set(selected.document.keywords[:5])
                
                if candidate_keywords and selected_keywords:
                    overlap = len(candidate_keywords & selected_keywords) / len(candidate_keywords | selected_keywords)
                    if overlap > self.search_config.diversity_threshold:
                        too_similar = True
                        break
            
            if not too_similar:
                filtered.append(candidate)
        
        return filtered
    
    def search(self, query: str, k: int = 10) -> List[EnhancedSearchResult]:
        """Enhanced search with multiple ranking factors."""
        if self.index is None or self.index.ntotal == 0:
            print("❌ No documents in index. Add documents first.")
            return []
        
        results = []
        
        # Generate query embedding
        embedding_result = self.embedding_generator.generate([query])
        query_embedding = np.array(embedding_result['data'][0]['embedding'], dtype=np.float32)
        
        # Determine search size (get more candidates for re-ranking)
        search_k = min(self.search_config.max_results, max(k * 2, 20))
        
        # Semantic search
        scores, indices = self._compute_similarity_scores(query_embedding, search_k)
        
        # Keyword search (if hybrid enabled)
        keyword_results = {}
        if self.search_config.enable_hybrid:
            keyword_hits = self._keyword_search(query, search_k)
            keyword_results = {idx: score for idx, score in keyword_hits}
        
        # Build initial results
        for rank, (idx, semantic_score) in enumerate(zip(indices[0], scores[0])):
            if idx >= 0 and semantic_score >= self.search_config.min_score_threshold:
                # Get keyword score if available
                keyword_score = keyword_results.get(idx, 0.0)
                
                result = EnhancedSearchResult(
                    document=self.documents[idx],
                    semantic_score=float(semantic_score),
                    keyword_score=keyword_score,
                    rank=rank + 1
                )
                results.append(result)
        
        # If no re-ranking, set final scores equal to semantic scores
        if not self.search_config.enable_reranking:
            for result in results:
                if self.search_config.enable_hybrid:
                    result.final_score = (
                        self.search_config.semantic_weight * result.semantic_score +
                        self.search_config.keyword_weight * result.keyword_score
                    )
                else:
                    result.final_score = result.semantic_score
        
        # Re-rank results with multiple factors
        results = self._rerank_results(results, query)
        
        # Apply diversity filtering
        if len(results) > k:
            results = self._filter_diverse_results(results[:k * 2])
        
        # Return top k results
        return results[:k]
    
    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Save enhanced index with all metadata."""
        if self.index is None:
            raise ValueError("No index to save. Add documents first.")
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Prepare metadata
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.meta.json'))
        
        metadata = {
            'documents': [doc.to_dict() for doc in self.documents],
            'embedding_config': asdict(self.embedding_config),
            'search_config': asdict(self.search_config),
            'dimension': self.dimension,
            'version': '2.0_enhanced',
            'features': {
                'chunking_enabled': self.chunker is not None,
                'hybrid_search': self.search_config.enable_hybrid,
                'reranking': self.search_config.enable_reranking,
                'tfidf_available': self.tfidf_vectorizer is not None
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved enhanced index to {index_path}")
        print(f"✅ Saved metadata to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """Load enhanced index with all metadata."""
        if not Path(index_path).exists():
            raise FileNotFoundError(
                f"Index file not found: {index_path}\n"
                f"💡 Create an index first with:\n"
                f"   python improved_search.py --build-with-chunks --index-path {index_path}\n"
                f"   OR: python improved_search.py --build-sample --index-path {index_path}"
            )
        
        # Load metadata
        if metadata_path is None:
            metadata_path = str(Path(index_path).with_suffix('.meta.json'))
        
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Restore documents
        self.documents = [EnhancedDocument.from_dict(doc_data) for doc_data in metadata['documents']]
        self.dimension = metadata['dimension']
        
        # Load search config if available
        if 'search_config' in metadata:
            search_config_data = metadata['search_config']
            self.search_config = SearchConfig(**search_config_data)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Rebuild TF-IDF if hybrid search is enabled
        if self.search_config.enable_hybrid and TfidfVectorizer:
            self._build_tfidf_index()
        
        version = metadata.get('version', '1.0')
        features = metadata.get('features', {})
        
        print(f"✅ Loaded enhanced index from {index_path} (v{version})")
        print(f"📚 Index contains {len(self.documents)} documents")
        
        if features:
            print("🎯 Enhanced features:")
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {feature.replace('_', ' ').title()}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        if not self.documents:
            return {}
        
        # Basic stats
        stats = {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'embedding_provider': self.embedding_config.provider,
            'embedding_model': self.embedding_config.model
        }
        
        # Enhanced stats
        chunks = sum(1 for doc in self.documents if doc.parent_document)
        original_docs = len(self.documents) - chunks
        
        stats.update({
            'original_documents': original_docs,
            'chunks_created': chunks,
            'avg_keywords_per_doc': sum(len(doc.keywords) for doc in self.documents) / len(self.documents),
            'documents_with_dates': sum(1 for doc in self.documents if doc.created_date),
            'search_features': {
                'hybrid_search': self.search_config.enable_hybrid,
                'reranking': self.search_config.enable_reranking,
                'query_expansion': self.search_config.enable_query_expansion,
                'similarity_metric': self.search_config.similarity_metric
            }
        })
        
        # Chunking stats
        if chunks > 0:
            chunk_docs = [doc for doc in self.documents if doc.parent_document]
            chunk_sizes = [doc.chunk_info.get('char_count', 0) for doc in chunk_docs if doc.chunk_info]
            
            if chunk_sizes:
                stats['chunking_stats'] = {
                    'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
                    'min_chunk_size': min(chunk_sizes),
                    'max_chunk_size': max(chunk_sizes),
                    'strategies_used': list(set(
                        doc.chunk_info.get('strategy', 'unknown') 
                        for doc in chunk_docs if doc.chunk_info
                    ))
                }
        
        return stats


# Utility functions for creating enhanced documents
def create_enhanced_document_from_text(text: str, doc_id: str, title: str = None, 
                                     metadata: Dict[str, Any] = None) -> EnhancedDocument:
    """Create an enhanced document from text."""
    return EnhancedDocument(
        id=doc_id,
        title=title or doc_id,
        content=text,
        metadata=metadata or {},
        created_date=datetime.now()
    )


def load_documents_from_directory(directory: Path) -> List[EnhancedDocument]:
    """Load documents from directory as enhanced documents."""
    documents = []
    
    for file_path in directory.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                doc = EnhancedDocument(
                    id=str(file_path.relative_to(directory)),
                    title=file_path.stem,
                    content=content,
                    metadata={
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_type': 'text'
                    },
                    created_date=datetime.fromtimestamp(file_path.stat().st_mtime)
                )
                documents.append(doc)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
    
    return documents


def create_sample_enhanced_documents() -> List[EnhancedDocument]:
    """Create sample enhanced documents for demonstration."""
    
    sample_docs = [
        {
            "id": "ai_intro_2024",
            "title": "Introduction to Artificial Intelligence (2024 Edition)",
            "content": "Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. Modern AI systems have evolved significantly since 2020, with breakthrough developments in large language models, computer vision, and autonomous systems. Key applications include natural language processing, image recognition, autonomous vehicles, and intelligent assistants.",
            "metadata": {"category": "AI Basics", "difficulty": "beginner", "year": 2024, "priority": "high"}
        },
        {
            "id": "machine_learning_advanced", 
            "title": "Advanced Machine Learning Techniques",
            "content": "Machine Learning (ML) encompasses sophisticated algorithms that learn patterns from data without explicit programming. Advanced techniques include ensemble methods like Random Forests and Gradient Boosting, deep learning architectures such as Transformers and Convolutional Neural Networks, and specialized approaches like reinforcement learning and federated learning. These methods excel in complex tasks like natural language understanding, computer vision, and strategic game playing.",
            "metadata": {"category": "Machine Learning", "difficulty": "advanced", "year": 2024, "priority": "high"}
        },
        {
            "id": "deep_learning_trends",
            "title": "Deep Learning and Neural Networks: Current Trends",
            "content": "Deep Learning continues to revolutionize AI with architectures like Transformers dominating natural language processing and computer vision. Recent trends include attention mechanisms, self-supervised learning, and large-scale pre-trained models. Key developments include GPT models for text generation, Vision Transformers for image processing, and multimodal models that understand both text and images. The field is moving towards more efficient architectures and better interpretability.",
            "metadata": {"category": "Deep Learning", "difficulty": "expert", "year": 2024, "priority": "high"}
        },
        {
            "id": "llm_deployment",
            "title": "Large Language Model Deployment Strategies",
            "content": "Deploying Large Language Models (LLMs) in production requires careful consideration of computational resources, latency requirements, and cost optimization. Strategies include model quantization, distillation, and efficient inference frameworks like vLLM and TensorRT. Cloud deployment options range from managed services like Azure OpenAI to custom infrastructure using Kubernetes and specialized hardware like GPUs and TPUs. Edge deployment is emerging for privacy-sensitive applications.",
            "metadata": {"category": "MLOps", "difficulty": "expert", "year": 2024, "priority": "medium"}
        },
        {
            "id": "ai_ethics_2024",
            "title": "AI Ethics and Responsible AI Development",
            "content": "As AI systems become more powerful and prevalent, ethical considerations are paramount. Key principles include fairness, accountability, transparency, and privacy. Responsible AI development involves bias detection and mitigation, explainable AI techniques, and robust testing frameworks. Emerging regulations like the EU AI Act are shaping industry practices. Organizations must implement AI governance frameworks and ethical review processes.",
            "metadata": {"category": "AI Ethics", "difficulty": "intermediate", "year": 2024, "priority": "high"}
        },
        {
            "id": "vector_search_advanced",
            "title": "Advanced Vector Search and Embeddings",
            "content": "Vector search has become the foundation of modern semantic search and retrieval-augmented generation (RAG) systems. Advanced techniques include hybrid search combining dense and sparse vectors, approximate nearest neighbor algorithms like HNSW and IVF, and specialized vector databases like Pinecone, Weaviate, and Qdrant. Embedding models have evolved from Word2Vec to transformer-based models like BERT and specialized embedding models like E5 and BGE.",
            "metadata": {"category": "Search", "difficulty": "advanced", "year": 2024, "priority": "medium"}
        },
        {
            "id": "rag_systems_production",
            "title": "Production RAG Systems: Best Practices",
            "content": "Retrieval Augmented Generation (RAG) systems in production require sophisticated engineering approaches. Best practices include chunking strategies for optimal context windows, hybrid retrieval combining semantic and keyword search, re-ranking mechanisms, and query expansion techniques. Production considerations include latency optimization, caching strategies, vector index management, and monitoring for hallucinations and retrieval quality.",
            "metadata": {"category": "RAG", "difficulty": "expert", "year": 2024, "priority": "high"}
        }
    ]
    
    enhanced_docs = []
    for i, doc_data in enumerate(sample_docs):
        doc = EnhancedDocument(
            **doc_data,
            created_date=datetime(2024, 4, 1 + i),  # Spread across April 2024
            boost_factor=1.5 if doc_data["metadata"].get("priority") == "high" else 1.0
        )
        enhanced_docs.append(doc)
    
    return enhanced_docs


def print_enhanced_search_results(results: List[EnhancedSearchResult], show_content: bool = False, 
                                show_explanation: bool = False) -> None:
    """Print enhanced search results with scoring details."""
    if not results:
        print("❌ No results found.")
        return
    
    print(f"\n🔍 Found {len(results)} enhanced results:\n")
    
    for result in results:
        doc = result.document
        print(f"#{result.rank} 📄 {doc.title}")
        print(f"   🆔 ID: {doc.id}")
        print(f"   📊 Final Score: {result.final_score:.4f}")
        
        # Show score breakdown
        if show_explanation and result.explanation:
            print(f"   🎯 Score Breakdown:")
            print(f"      Semantic: {result.semantic_score:.3f}")
            if hasattr(result, 'keyword_score') and result.keyword_score > 0:
                print(f"      Keyword: {result.keyword_score:.3f}")
            if result.recency_score > 1.0:
                print(f"      Recency boost: {result.recency_score:.2f}x")
            if result.boost_score > 1.0:
                print(f"      Manual boost: {result.boost_score:.2f}x")
        
        # Show metadata
        if doc.metadata:
            relevant_meta = {k: v for k, v in doc.metadata.items() 
                           if k in ['category', 'difficulty', 'year', 'priority']}
            if relevant_meta:
                meta_str = ', '.join([f"{k}: {v}" for k, v in relevant_meta.items()])
                print(f"   🏷️  Metadata: {meta_str}")
        
        # Show chunking info if available
        if doc.chunk_info and doc.parent_document:
            chunk_info = doc.chunk_info
            print(f"   ✂️  Chunk: {chunk_info.get('chunk_index', 'N/A')} "
                  f"({chunk_info.get('char_count', 'N/A')} chars, "
                  f"strategy: {chunk_info.get('strategy', 'unknown')})")
            print(f"   📄 Parent: {doc.parent_document}")
        
        # Show keywords
        if doc.keywords:
            keywords_str = ', '.join(doc.keywords[:5])
            print(f"   🔑 Keywords: {keywords_str}")
        
        # Show content preview
        if show_content:
            content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
            print(f"   📖 Content: {content_preview}")
        
        print()


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced semantic search with improved ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build enhanced index with chunking
  python improved_search.py --build-with-chunks --input docs/ --index-path enhanced.faiss
  
  # Search with re-ranking and hybrid mode
  python improved_search.py --query "machine learning" --hybrid --rerank --index-path enhanced.faiss
  
  # Interactive mode with all features
  python improved_search.py --interactive --index-path enhanced.faiss
  
  # Performance comparison
  python improved_search.py --compare --query "AI deployment" --index-path enhanced.faiss
"""
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--build-sample', action='store_true',
                             help='Build index from enhanced sample documents')
    action_group.add_argument('--build-from-dir', type=Path, metavar='DIR',
                             help='Build index from text files in directory')
    action_group.add_argument('--build-with-chunks', action='store_true',
                             help='Build sample index using chunking')
    action_group.add_argument('--query', type=str,
                             help='Search query string')
    action_group.add_argument('--interactive', action='store_true',
                             help='Run interactive search mode')
    action_group.add_argument('--stats', action='store_true',
                             help='Show enhanced index statistics')
    action_group.add_argument('--compare', action='store_true',
                             help='Compare search strategies')
    
    # Index arguments
    parser.add_argument('--index-path', type=str, default='enhanced_search.faiss',
                       help='Path to enhanced FAISS index file')
    parser.add_argument('--input', type=Path,
                       help='Input directory for building index')
    
    # Search configuration
    parser.add_argument('--hybrid', action='store_true', 
                       help='Enable hybrid semantic + keyword search')
    parser.add_argument('--rerank', action='store_true',
                       help='Enable re-ranking with multiple factors')
    parser.add_argument('--boost-recent', action='store_true',
                       help='Boost recently created documents')
    parser.add_argument('--similarity', choices=['cosine', 'dot_product'],
                       default='cosine', help='Similarity metric')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Weight for semantic search in hybrid mode')
    
    # Chunking options
    parser.add_argument('--chunk-strategy', choices=['fixed', 'sentence', 'semantic', 'hybrid'],
                       default='fixed', help='Chunking strategy')
    parser.add_argument('--chunk-size', type=int, default=1500,
                       help='Chunk size for better ranking (default: 1500)')
    parser.add_argument('--overlap', type=int, default=300,
                       help='Chunk overlap (default: 300)') 
    
    # Output options
    parser.add_argument('-k', '--top-k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--show-content', action='store_true',
                       help='Show document content in results')
    parser.add_argument('--show-explanation', action='store_true',
                       help='Show scoring explanation')
    
    # Provider arguments
    parser.add_argument('-p', '--provider', choices=['openai', 'azure'],
                       help='Embedding provider')
    parser.add_argument('-m', '--model', help='Embedding model name')
    
    return parser


def main() -> int:
    """Main function."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        # Configure embeddings
        provider = args.provider or _default_provider()
        model = args.model or _default_model(provider)
        
        embedding_config = EmbeddingConfig(
            provider=provider,
            model=model
        )
        
        # Configure enhanced search
        search_config = SearchConfig(
            similarity_metric=args.similarity,
            enable_hybrid=args.hybrid,
            semantic_weight=args.alpha,
            keyword_weight=1.0 - args.alpha,
            enable_reranking=args.rerank,
            boost_recent=args.boost_recent
        )
        
        # Initialize enhanced search
        search_system = EnhancedSemanticSearch(embedding_config, search_config)
        
        # Configure chunking if requested
        if args.build_with_chunks or any(arg for arg in ['chunk_strategy', 'chunk_size'] 
                                        if hasattr(args, arg)):
            from chunking import ChunkingStrategy
            chunking_config = ChunkingConfig(
                strategy=ChunkingStrategy(args.chunk_strategy) if hasattr(args, 'chunk_strategy') else ChunkingStrategy.FIXED,
                chunk_size=args.chunk_size,
                overlap_size=args.overlap
            )
            search_system.set_chunking_config(chunking_config)
        
        # Handle different actions
        if args.build_sample:
            print("📚 Creating enhanced sample documents...")
            documents = create_sample_enhanced_documents()
            search_system.add_documents(documents, use_chunking=False)
            search_system.save_index(args.index_path)
            
        elif args.build_from_dir:
            print(f"📂 Loading documents from {args.input or args.build_from_dir}...")
            doc_dir = args.input or args.build_from_dir
            documents = load_documents_from_directory(doc_dir)
            if not documents:
                print("❌ No documents found in directory.")
                return 1
            print(f"📚 Found {len(documents)} documents")
            search_system.add_documents(documents, use_chunking=False)
            search_system.save_index(args.index_path)
            
        elif args.build_with_chunks:
            print("📚 Creating enhanced sample documents with chunking...")
            documents = create_sample_enhanced_documents()
            search_system.add_documents(documents, use_chunking=True)
            search_system.save_index(args.index_path)
            
        elif args.query:
            print(f"📖 Loading enhanced index from {args.index_path}")
            search_system.load_index(args.index_path)
            
            print(f"🔍 Searching for: '{args.query}'")
            results = search_system.search(args.query, args.top_k)
            print_enhanced_search_results(results, args.show_content, args.show_explanation)
            
        elif args.interactive:
            print(f"📖 Loading enhanced index from {args.index_path}")
            search_system.load_index(args.index_path)
            
            print(f"\n🧠 Enhanced Semantic Search - Interactive Mode")
            print("💡 Type 'quit' to exit, 'help' for commands")
            print("-" * 60)
            
            while True:
                try:
                    query = input("\n🔍 Search query: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if query.lower() == 'help':
                        print("\n📋 Advanced commands:")
                        print("  <query>           - Search with current settings")
                        print("  hybrid <query>    - Search with hybrid mode")
                        print("  plain <query>     - Search without enhancements") 
                        print("  stats             - Show index statistics")
                        print("  help              - Show this help")
                        print("  quit              - Exit")
                        continue
                    
                    if query.lower() == 'stats':
                        stats = search_system.get_stats()
                        print("\n📊 Enhanced Index Statistics:")
                        for key, value in stats.items():
                            if isinstance(value, dict):
                                print(f"  {key}:")
                                for subkey, subvalue in value.items():
                                    print(f"    {subkey}: {subvalue}")
                            else:
                                print(f"  {key}: {value}")
                        continue
                    
                    # Handle special command prefixes
                    original_hybrid = search_system.search_config.enable_hybrid
                    show_explanation = True
                    
                    if query.lower().startswith('hybrid '):
                        search_system.search_config.enable_hybrid = True
                        query = query[7:]
                        print("🔄 Using hybrid search mode")
                    elif query.lower().startswith('plain '):
                        search_system.search_config.enable_hybrid = False
                        search_system.search_config.enable_reranking = False
                        query = query[6:]
                        show_explanation = False
                        print("⚡ Using plain semantic search")
                    
                    if not query:
                        continue
                    
                    # Perform search
                    results = search_system.search(query, 5)
                    print_enhanced_search_results(results, show_content=True, 
                                               show_explanation=show_explanation)
                    
                    # Restore original settings
                    search_system.search_config.enable_hybrid = original_hybrid
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Search error: {e}")
            
        elif args.stats:
            print(f"📖 Loading enhanced index from {args.index_path}")
            search_system.load_index(args.index_path)
            
            stats = search_system.get_stats()
            print("\n📊 Enhanced Index Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        
        elif args.compare:
            # TODO: Implement comparison between different search strategies
            print("🔄 Strategy comparison not yet implemented")
            return 0
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())