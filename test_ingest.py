#!/usr/bin/env python3
"""
Quick Test: Document Processing with LangChain Loaders
=====================================================

Demonstrates the document processing pipeline with sample files.
Tests LangChain loader integration and chunking strategies.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

try:
    from ingest import LangChainDocumentProcessor, ProcessingConfig, DocumentType
    print("✅ Import successful: Document processor ready")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("📦 Install dependencies: pip install langchain langchain-community")
    sys.exit(1)

def create_sample_documents():
    """Create sample documents for testing."""
    test_dir = Path("test_docs")
    test_dir.mkdir(exist_ok=True)
    
    # Sample text document
    text_content = """
    Enterprise AI Assistant Documentation
    ===================================
    
    This document describes the features and capabilities of the Enterprise AI Assistant.
    
    Key Features:
    1. Document Processing - Support for multiple file formats
    2. Vector Search - FAISS-based similarity search  
    3. Hybrid Retrieval - Combines semantic and keyword search
    4. LLM Integration - Azure OpenAI for response generation
    5. Chunking Strategies - Optimized for different document types
    
    Processing Pipeline:
    The system follows a multi-stage processing pipeline:
    - Document loading with format-specific loaders
    - Content extraction and cleaning
    - Intelligent chunking with overlap
    - Metadata enrichment and indexing
    - Vector embedding generation
    
    Performance Characteristics:
    - Processing speed: 50+ documents/minute
    - Chunk size: 1000 characters (configurable)
    - Overlap: 20% for context preservation
    - Supported formats: PDF, DOCX, TXT, JSON, CSV, HTML, MD
    
    Quality Assurance:
    All processing includes comprehensive error handling, fallback mechanisms,
    and quality validation to ensure robust operation in production environments.
    """
    
    with open(test_dir / "sample_text.txt", "w") as f:
        f.write(text_content)
    
    # Sample JSON data
    json_data = {
        "documents": [
            {
                "id": "DOC001",
                "title": "Insurance Policy Overview",
                "content": "This policy provides comprehensive coverage for property damage, liability, and personal protection. Coverage includes dwelling protection up to $500,000, personal property coverage up to $250,000, and liability protection up to $1,000,000.",
                "type": "policy",
                "category": "insurance"
            },
            {
                "id": "DOC002", 
                "title": "Claims Processing Guide",
                "content": "To file a claim, contact our 24/7 claims hotline at 1-800-CLAIMS. You will need your policy number, incident date, and detailed description of the damage or loss. Claims are typically processed within 5-7 business days.",
                "type": "guide",
                "category": "claims"
            }
        ]
    }
    
    with open(test_dir / "sample_data.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    # Sample CSV data
    csv_content = """name,type,coverage,premium
Auto Insurance,Vehicle,Collision and Comprehensive,1200
Home Insurance,Property,Dwelling and Personal Property,850
Life Insurance,Personal,Term Life Coverage,480
Health Insurance,Medical,Medical and Prescription,2400"""
    
    with open(test_dir / "sample_data.csv", "w") as f:
        f.write(csv_content)
    
    print(f"📁 Created sample documents in {test_dir}/")
    return test_dir

def test_single_file_processing():
    """Test processing a single file."""
    print("\n🔬 Test 1: Single File Processing")
    print("=" * 50)
    
    # Create test configuration
    config = ProcessingConfig(
        chunk_size=500,  # Smaller for testing
        chunk_overlap=100,
        chunking_strategy="recursive",
        max_workers=2,
        output_format="json"
    )
    
    processor = LangChainDocumentProcessor(config)
    test_dir = create_sample_documents()
    
    # Process text file
    text_file = test_dir / "sample_text.txt"
    output_dir = Path("test_output")
    
    chunks = processor.process_file(text_file, output_dir)
    
    print(f"📄 Processed: {text_file.name}")
    print(f"📊 Generated: {len(chunks)} chunks")
    
    if chunks:
        sample = chunks[0]
        print(f"\n📋 Sample chunk (ID: {sample.chunk_id}):")
        print("-" * 40)
        print(sample.content[:200] + "..." if len(sample.content) > 200 else sample.content)
        print("-" * 40)
        print(f"🏷️  Metadata: {sample.metadata}")
    
    return chunks

def test_multiple_formats():
    """Test processing different file formats."""
    print("\n🔬 Test 2: Multiple File Formats")  
    print("=" * 50)
    
    config = ProcessingConfig(
        chunk_size=300,
        chunk_overlap=50,
        output_format="json"
    )
    
    processor = LangChainDocumentProcessor(config)
    test_dir = Path("test_docs")
    output_dir = Path("test_output")
    
    # Test each file type
    test_files = [
        ("sample_text.txt", DocumentType.TXT),
        ("sample_data.json", DocumentType.JSON),
        ("sample_data.csv", DocumentType.CSV)
    ]
    
    results = {}
    
    for filename, expected_type in test_files:
        file_path = test_dir / filename
        if file_path.exists():
            print(f"\n📁 Processing: {filename}")
            
            # Test type detection
            detected_type = processor.detect_document_type(file_path)
            print(f"🔍 Detected type: {detected_type}")
            print(f"✅ Expected type: {expected_type}")
            
            # Process file
            chunks = processor.process_file(file_path, output_dir)
            results[filename] = {
                'chunks': len(chunks),
                'detected_type': detected_type,
                'success': len(chunks) > 0
            }
            
            print(f"📊 Result: {len(chunks)} chunks generated")
    
    return results

def test_chunking_strategies():
    """Test different chunking strategies."""
    print("\n🔬 Test 3: Chunking Strategies")
    print("=" * 50)
    
    strategies = ["recursive", "character"]
    test_file = Path("test_docs/sample_text.txt")
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Testing strategy: {strategy}")
        
        config = ProcessingConfig(
            chunk_size=400,
            chunk_overlap=80,
            chunking_strategy=strategy
        )
        
        processor = LangChainDocumentProcessor(config)
        chunks = processor.process_file(test_file, Path("test_output"))
        
        results[strategy] = {
            'chunk_count': len(chunks),
            'avg_length': sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            'chunks': chunks
        }
        
        print(f"📊 Chunks: {len(chunks)}")
        print(f"📏 Avg length: {results[strategy]['avg_length']:.1f} chars")
    
    return results

def test_batch_processing():
    """Test batch processing of multiple files."""
    print("\n🔬 Test 4: Batch Processing")
    print("=" * 50)
    
    config = ProcessingConfig(
        chunk_size=500,
        chunk_overlap=100,
        max_workers=2,
        output_format="json"
    )
    
    processor = LangChainDocumentProcessor(config)
    test_dir = Path("test_docs")
    output_dir = Path("test_output_batch")
    
    # Process entire directory
    start_time = datetime.now()
    all_chunks = processor.process_directory(test_dir, output_dir)
    end_time = datetime.now()
    
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"📁 Processed directory: {test_dir}")
    print(f"📊 Total chunks: {len(all_chunks)}")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    
    # Check summary file
    summary_file = output_dir / "processing_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"📋 Files processed: {summary['processed_files']}")
        print(f"🎯 Summary saved: {summary_file}")
    
    return all_chunks

def run_comprehensive_test():
    """Run all tests and display results."""
    print("🚀 LangChain Document Processing Tests")
    print("=" * 60)
    
    try:
        # Test 1: Single file processing  
        chunks1 = test_single_file_processing()
        
        # Test 2: Multiple file formats
        results2 = test_multiple_formats()
        
        # Test 3: Chunking strategies
        results3 = test_chunking_strategies()
        
        # Test 4: Batch processing
        chunks4 = test_batch_processing()
        
        # Summary
        print("\n🎯 Test Summary")
        print("=" * 50)
        print(f"✅ Single file: {len(chunks1)} chunks")
        print(f"✅ Multiple formats: {len(results2)} files tested")
        print(f"✅ Chunking strategies: {len(results3)} strategies tested")
        print(f"✅ Batch processing: {len(chunks4)} total chunks")
        
        print("\n📊 Format Processing Results:")
        for filename, result in results2.items():
            status = "✅" if result['success'] else "❌"
            print(f"{status} {filename}: {result['chunks']} chunks ({result['detected_type']})")
        
        print("\n📏 Chunking Strategy Comparison:")
        for strategy, result in results3.items():
            print(f"🔄 {strategy}: {result['chunk_count']} chunks, avg {result['avg_length']:.0f} chars")
        
        print("\n🎉 All tests completed successfully!")
        print("📁 Check 'test_output/' and 'test_output_batch/' for results")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test()