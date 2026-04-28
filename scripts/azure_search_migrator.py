#!/usr/bin/env python3
"""
Azure AI Search Migration Tool
=============================

Migrates existing FAISS-based RAG system to Azure AI Search with:
- Vector search capabilities
- Hybrid text + vector search
- Rich metadata indexing
- Integration with existing LangChain document processing
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
import logging

# Add paths for existing components
sys.path.append(str(Path(__file__).parent.parent / "apps" / "rag"))
sys.path.append(str(Path(__file__).parent))

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        ComplexField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile,
        SemanticConfiguration,
        SemanticSearch,
        SemanticPrioritizedFields,
        SemanticField
    )
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    print("❌ Azure Search SDK not installed. Run: pip install azure-search-documents azure-identity")
    AZURE_SEARCH_AVAILABLE = False
    sys.exit(1)

# Import your existing components
from ingest import LangChainDocumentProcessor, ProcessedChunk
from embeddings import EmbeddingGenerator, EmbeddingConfig
from improved_search import EnhancedSemanticSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureSearchConfig:
    """Configuration for Azure AI Search integration."""
    
    def __init__(self):
        self.service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
        self.query_key = os.getenv("AZURE_SEARCH_QUERY_KEY")
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        
        if not all([self.service_name, self.admin_key, self.endpoint]):
            raise ValueError("Missing Azure Search configuration. Check your .env file.")
        
        # Index configuration
        self.index_name = "enterprise-rag-index"
        self.vector_config_name = "enterprise-vector-config"
        self.semantic_config_name = "enterprise-semantic-config"

class AzureSearchMigrator:
    """Migrates FAISS-based RAG to Azure AI Search."""
    
    def __init__(self):
        self.config = AzureSearchConfig()
        
        # Initialize Azure Search clients
        credential = AzureKeyCredential(self.config.admin_key)
        self.index_client = SearchIndexClient(
            endpoint=self.config.endpoint,
            credential=credential
        )
        self.search_client = SearchClient(
            endpoint=self.config.endpoint,
            index_name=self.config.index_name,
            credential=credential
        )
        
        # Initialize embedding service for vector generation
        embedding_config = EmbeddingConfig(
            provider="azure",
            model="text-embedding-3-small"
        )
        self.embedding_service = EmbeddingGenerator(embedding_config)
        
    def create_search_index(self) -> bool:
        """Create Azure AI Search index with vector and semantic search."""
        try:
            # Define vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="enterprise-hnsw",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name=self.config.vector_config_name,
                        algorithm_configuration_name="enterprise-hnsw"
                    )
                ]
            )
            
            # Define semantic search configuration
            semantic_search = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=self.config.semantic_config_name,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="content")],
                            keywords_fields=[SemanticField(field_name="keywords")]
                        )
                    )
                ]
            )
            
            # Define index fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="contentVector", 
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, 
                    vector_search_dimensions=1536,
                    vector_search_profile_name=self.config.vector_config_name
                ),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="chunkIndex", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="createdAt", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="processingTime", type=SearchFieldDataType.Double, filterable=True, sortable=True),
                SimpleField(name="checksum", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="fileType", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="fileSize", type=SearchFieldDataType.Int64, filterable=True, sortable=True),
                SimpleField(name="pageCount", type=SearchFieldDataType.Int32, filterable=True),
                SearchableField(name="keywords", type=SearchFieldDataType.String),
                SimpleField(name="tokens", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            ]
            
            # Create the index
            index = SearchIndex(
                name=self.config.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            # Create or update the index
            result = self.index_client.create_or_update_index(index)
            logger.info(f"✅ Created Azure AI Search index: {result.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create search index: {e}")
            return False
    
    def migrate_chunks_to_azure_search(self, chunks_file: Path) -> bool:
        """Migrate processed chunks to Azure AI Search."""
        try:
            # Load processed chunks
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            logger.info(f"📄 Loaded {len(chunks_data)} chunks from {chunks_file}")
            
            def convert_to_iso_datetime(date_str: str) -> str:
                """Convert Python datetime string to ISO 8601 format for Azure Search"""
                if not date_str:
                    return None
                try:
                    from datetime import datetime
                    # Parse the datetime string (assuming format: "2026-04-16 10:33:56.095812")
                    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
                    # Return in ISO 8601 format
                    return dt.isoformat() + 'Z'
                except Exception as e:
                    logger.warning(f"Could not parse datetime '{date_str}': {e}")
                    return None
            
            # Convert chunks to Azure Search documents
            documents = []
            for chunk_data in chunks_data:
                # Generate vector embedding for content
                content = chunk_data['content']
                embedding_response = self.embedding_service.generate([content])
                vector = embedding_response['data'][0]['embedding']
                
                # Create Azure Search document with proper datetime conversion
                doc = {
                    "id": chunk_data['chunk_id'],
                    "content": content,
                    "contentVector": vector,
                    "title": chunk_data['metadata'].get('title', ''),
                    "source": chunk_data['source_metadata']['source_path'],
                    "chunkIndex": chunk_data['chunk_index'],
                    "createdAt": convert_to_iso_datetime(chunk_data['source_metadata']['created_at']),
                    "processingTime": chunk_data['source_metadata']['processing_time'],
                    "checksum": chunk_data['source_metadata']['checksum'],
                    "fileType": chunk_data['source_metadata']['file_type'],
                    "fileSize": chunk_data['source_metadata']['file_size'],
                    "pageCount": chunk_data['source_metadata'].get('page_count'),
                    "keywords": ', '.join(chunk_data['metadata'].get('keywords', [])),
                    "tokens": chunk_data['tokens']
                }
                documents.append(doc)
            
            # Upload documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = self.search_client.upload_documents(batch)
                logger.info(f"📤 Uploaded batch {i//batch_size + 1}: {len(batch)} documents")
            
            logger.info(f"✅ Successfully migrated {len(documents)} documents to Azure AI Search")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate chunks: {e}")
            return False
    
    def test_search_capabilities(self) -> None:
        """Test Azure AI Search vector and hybrid search."""
        test_queries = [
            "machine learning deployment",
            "artificial intelligence ethics", 
            "vector search algorithms"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing query: '{query}'")
            
            # Generate query vector
            embedding_response = self.embedding_service.generate([query])
            query_vector = embedding_response['data'][0]['embedding']
            
            # Test vector search
            vector_results = self.search_client.search(
                search_text=None,
                vector_queries=[{
                    "vector": query_vector,
                    "k_nearest_neighbors": 3,
                    "fields": "contentVector"
                }],
                select=["id", "content", "source", "chunkIndex"],
                top=3
            )
            
            logger.info("📊 Vector Search Results:")
            for i, result in enumerate(vector_results, 1):
                logger.info(f"  #{i} Score: {result['@search.score']:.4f}")
                logger.info(f"      Content: {result['content'][:100]}...")
                logger.info(f"      Source: {result['source']}")
            
            # Test hybrid search (text + vector)
            hybrid_results = self.search_client.search(
                search_text=query,
                vector_queries=[{
                    "vector": query_vector,
                    "k_nearest_neighbors": 3,
                    "fields": "contentVector"
                }],
                select=["id", "content", "source", "chunkIndex"],
                top=3
            )
            
            logger.info("🔄 Hybrid Search Results:")
            for i, result in enumerate(hybrid_results, 1):
                logger.info(f"  #{i} Score: {result['@search.score']:.4f}")
                logger.info(f"      Content: {result['content'][:100]}...")
                logger.info(f"      Source: {result['source']}")

def main():
    """Main migration workflow."""
    logger.info("🚀 Starting Azure AI Search Migration")
    
    migrator = AzureSearchMigrator()
    
    # Step 1: Create search index
    logger.info("📋 Step 1: Creating Azure AI Search index...")
    if not migrator.create_search_index():
        logger.error("❌ Index creation failed. Exiting.")
        return
    
    # Step 2: Find existing chunks files to migrate
    chunks_dir = Path("chunks")
    if not chunks_dir.exists():
        logger.error("❌ Chunks directory not found. Run document processing first.")
        return
    
    chunks_files = list(chunks_dir.glob("*_chunks.json"))
    if not chunks_files:
        logger.error("❌ No chunk files found. Run document processing first.")
        return
    
    # Step 3: Migrate chunks to Azure Search
    logger.info(f"📄 Step 2: Migrating {len(chunks_files)} chunk files...")
    for chunks_file in chunks_files:
        logger.info(f"📤 Migrating: {chunks_file}")
        migrator.migrate_chunks_to_azure_search(chunks_file)
    
    # Step 4: Test search capabilities
    logger.info("🧪 Step 3: Testing search capabilities...")
    migrator.test_search_capabilities()
    
    logger.info("🎉 Azure AI Search migration completed!")

if __name__ == "__main__":
    main()