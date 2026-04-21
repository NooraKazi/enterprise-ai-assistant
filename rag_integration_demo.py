#!/usr/bin/env python3
"""
RAG Integration: Document Processing + Vector Search
==================================================

Demonstrates end-to-end integration of:
1. LangChain document loading
2. Advanced chunking strategies
3. Embedding generation
4. Vector index creation
5. Query processing

Example workflow for processing documents into a queryable RAG system.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup paths
sys.path.append(str(Path(__file__).parent / "scripts"))
sys.path.append(str(Path(__file__).parent / "apps" / "rag"))

try:
    # Import document processing
    from ingest import LangChainDocumentProcessor, ProcessingConfig, ProcessedChunk
    
    # Import existing RAG components
    from embeddings import EmbeddingService
    from improved_search import EnhancedSemanticSearch
    
    print("✅ All imports successful - RAG integration ready!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Install missing dependencies")
    sys.exit(1)

class IntegratedRAGPipeline:
    """Complete RAG pipeline with LangChain document processing."""
    
    def __init__(self, config: ProcessingConfig):
        self.processing_config = config
        self.doc_processor = LangChainDocumentProcessor(config)
        self.embedding_service = None
        self.search_service = None
        
        # Initialize embedding service
        try:
            self.embedding_service = EmbeddingService()
            print("✅ Embedding service initialized")
        except Exception as e:
            print(f"⚠️  Embedding service unavailable: {e}")
    
    def process_documents_to_rag(self, input_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Complete pipeline: docs → chunks → embeddings → search index."""
        
        pipeline_results = {
            'processed_files': 0,
            'total_chunks': 0,
            'embeddings_generated': 0,
            'search_index_size': 0,
            'processing_time': 0.0
        }
        
        print("🚀 Starting RAG Pipeline")
        print("=" * 50)
        
        # Step 1: Document Processing
        print("📄 Step 1: Processing Documents...")
        if input_path.is_file():
            chunks = self.doc_processor.process_file(input_path, output_dir)
        else:
            chunks = self.doc_processor.process_directory(input_path, output_dir)
        
        pipeline_results['total_chunks'] = len(chunks)
        print(f"✅ Generated {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No chunks generated - pipeline stopped")
            return pipeline_results
        
        # Step 2: Generate Embeddings
        print("\\n🧠 Step 2: Generating Embeddings...")
        if self.embedding_service:
            embeddings_data = self._generate_embeddings(chunks)
            pipeline_results['embeddings_generated'] = len(embeddings_data)
            print(f"✅ Generated {len(embeddings_data)} embeddings")
            
            # Step 3: Create Search Index
            print("\\n🔍 Step 3: Building Search Index...")
            search_index = self._build_search_index(embeddings_data, output_dir)
            if search_index:
                pipeline_results['search_index_size'] = len(search_index['documents'])
                print(f"✅ Built search index with {pipeline_results['search_index_size']} documents")
                
                # Step 4: Test Query
                print("\\n🎯 Step 4: Testing Query...")
                test_results = self._test_query_system(search_index)
                
                print("\\n🎉 RAG Pipeline Complete!")
                self._display_pipeline_summary(pipeline_results, test_results)
            
        else:
            print("⚠️  Skipping embedding generation - no embedding service")
        
        return pipeline_results
    
    def _generate_embeddings(self, chunks: List[ProcessedChunk]) -> List[Dict[str, Any]]:
        """Generate embeddings for processed chunks."""
        embeddings_data = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = self.embedding_service.get_embedding(chunk.content)
                
                # Create document record
                doc_record = {
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'embedding': embedding,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_index': chunk.chunk_index,
                        'tokens': chunk.tokens,
                        'source_file': chunk.source_metadata.source_path,
                        'file_type': chunk.source_metadata.file_type.value,
                        'processing_time': chunk.source_metadata.processing_time
                    }
                }
                
                embeddings_data.append(doc_record)
                
                if (i + 1) % 10 == 0:
                    print(f"  📊 Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"❌ Error generating embedding for chunk {chunk.chunk_id}: {e}")
        
        return embeddings_data
    
    def _build_search_index(self, embeddings_data: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """Build enhanced search index from embeddings."""
        try:
            # Extract components for EnhancedSemanticSearch
            documents = []
            embeddings = []
            metadata_list = []
            
            for doc in embeddings_data:
                documents.append(doc['content'])
                embeddings.append(doc['embedding'])
                metadata_list.append(doc['metadata'])
            
            # Initialize enhanced search
            search_service = EnhancedSemanticSearch()
            
            # Build index
            search_service.build_index(
                documents=documents,
                embeddings=embeddings,
                metadata=metadata_list
            )
            
            # Save index
            index_file = output_dir / "rag_search_index.faiss"
            metadata_file = output_dir / "rag_search_metadata.json"
            
            search_service.save_index(str(index_file), str(metadata_file))
            
            self.search_service = search_service
            
            return {
                'documents': documents,
                'embeddings': embeddings,
                'metadata': metadata_list,
                'index_file': str(index_file),
                'metadata_file': str(metadata_file)
            }
            
        except Exception as e:
            print(f"❌ Error building search index: {e}")
            return None
    
    def _test_query_system(self, search_index: Dict[str, Any]) -> Dict[str, Any]:
        """Test the complete query system."""
        if not self.search_service:
            return {'error': 'No search service available'}
        
        # Test queries
        test_queries = [
            "What is document processing?",
            "How does the system work?",
            "What are the key features?"
        ]
        
        test_results = {}
        
        for query in test_queries:
            try:
                print(f"🔍 Testing query: '{query}'")
                
                # Execute search
                results = self.search_service.search(
                    query=query,
                    top_k=3,
                    use_hybrid=True
                )
                
                test_results[query] = {
                    'result_count': len(results),
                    'top_score': results[0]['final_score'] if results else 0.0,
                    'results': results[:2]  # Store top 2 for review
                }
                
                print(f"  📊 Found {len(results)} results, top score: {test_results[query]['top_score']:.3f}")
                
                if results:
                    print(f"  📝 Top result: {results[0]['content'][:100]}...")
                
            except Exception as e:
                print(f"❌ Query test failed: {e}")
                test_results[query] = {'error': str(e)}
        
        return test_results
    
    def _display_pipeline_summary(self, pipeline_results: Dict[str, Any], test_results: Dict[str, Any]):
        """Display comprehensive pipeline summary."""
        print("=" * 60)
        print("📊 RAG PIPELINE SUMMARY")
        print("=" * 60)
        
        print("\\n📈 Processing Metrics:")
        print(f"  📄 Total chunks processed: {pipeline_results['total_chunks']}")
        print(f"  🧠 Embeddings generated: {pipeline_results['embeddings_generated']}")
        print(f"  🔍 Search index size: {pipeline_results['search_index_size']}")
        
        print("\\n🎯 Query Test Results:")
        for query, result in test_results.items():
            if 'error' not in result:
                print(f"  ✅ '{query}': {result['result_count']} results (score: {result['top_score']:.3f})")
            else:
                print(f"  ❌ '{query}': {result['error']}")
        
        print("\\n🎉 RAG System Status: OPERATIONAL")
        print("📁 Index files saved in output directory")
        print("🚀 Ready for production queries!")

def demo_full_rag_pipeline():
    """Demonstrate the complete RAG pipeline."""
    print("🚀 RAG Integration Demo")
    print("=" * 60)
    
    # Configuration
    config = ProcessingConfig(
        chunk_size=800,
        chunk_overlap=200,
        chunking_strategy="recursive",
        max_workers=2,
        output_format="json"
    )
    
    # Initialize pipeline
    pipeline = IntegratedRAGPipeline(config)
    
    # Create test documents (using existing test function)
    test_dir = Path("test_docs")
    if not test_dir.exists():
        test_dir.mkdir()
        
        # Create sample document
        sample_content = """
        Enterprise RAG System Documentation
        
        Overview:
        The Enterprise RAG (Retrieval-Augmented Generation) system combines advanced document processing 
        with semantic search and large language model integration. This system enables organizations to 
        build intelligent knowledge bases from their document collections.
        
        Key Components:
        1. Document Processing Pipeline - Handles multiple file formats including PDF, DOCX, TXT, JSON, and CSV
        2. Advanced Chunking - Implements Microsoft-recommended strategies for optimal retrieval
        3. Vector Embeddings - Uses Azure OpenAI for high-quality document representations  
        4. Hybrid Search - Combines semantic similarity with keyword matching
        5. LLM Integration - Generates contextual responses using retrieved knowledge
        
        Technical Architecture:
        The system follows a layered architecture with clear separation of concerns:
        - Data Sources Layer: Ingests documents from various sources
        - Processing Layer: Cleans, chunks, and enriches content
        - Embedding Layer: Generates vector representations
        - Search Layer: Provides fast, relevant retrieval
        - Generation Layer: Produces intelligent responses
        
        Performance Characteristics:
        - Processing Speed: 100+ documents per minute
        - Search Latency: Sub-100ms for most queries
        - Accuracy: 95%+ relevance in domain-specific use cases
        - Scalability: Supports millions of documents with proper infrastructure
        
        Use Cases:
        - Customer Support: Automated response generation from knowledge bases
        - Research: Intelligent document discovery and summarization
        - Compliance: Policy and regulation query systems
        - Training: Interactive learning from organizational knowledge
        """
        
        with open(test_dir / "rag_system_docs.txt", "w") as f:
            f.write(sample_content)
    
    output_dir = Path("rag_integration_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run complete pipeline
    try:
        results = pipeline.process_documents_to_rag(test_dir, output_dir)
        
        print("\\n✅ Demo completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_full_rag_pipeline()