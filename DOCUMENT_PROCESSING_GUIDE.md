# Document Processing Guide 📚

Complete guide to using the enterprise-grade document processing system with LangChain loaders and advanced chunking strategies.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported File Types](#supported-file-types)
- [Processing Strategies](#processing-strategies)
- [Advanced Usage](#advanced-usage)
- [Integration with RAG](#integration-with-rag)

## Overview

The document processing system combines:
- **LangChain Document Loaders** - Support for 20+ file formats
- **Advanced Chunking** - Multiple strategies optimized for RAG
- **Metadata Extraction** - Rich document metadata and tracking
- **Batch Processing** - Parallel processing with progress tracking

## Installation

### 1. Install Dependencies
```bash
# Install enhanced dependencies
pip install -r requirements.txt

# Or install specific LangChain components
pip install langchain langchain-community langchain-text-splitters
pip install unstructured pytesseract pillow python-magic
```

### 2. Install Optional OCR Support
```bash
# For OCR on images and scanned PDFs
apt-get install tesseract-ocr  # Linux
brew install tesseract         # macOS
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Quick Start

### Process a Single File
```bash
# Basic processing
python scripts/ingest.py --input document.pdf --output chunks/

# With custom chunk size
python scripts/ingest.py --input document.pdf --chunk-size 1500 --chunk-overlap 300

# Verbose output with metadata
python scripts/ingest.py --input document.pdf --output chunks/ --verbose
```

### Process Multiple Files
```bash
# Process entire directory
python scripts/ingest.py --input-dir documents/ --output-dir processed/ --max-workers 8

# Process directory with specific settings
python scripts/ingest.py --input-dir documents/ --chunk-size 1000 --strategy recursive --output-format jsonl
```

### Interactive Mode
```bash
python scripts/ingest.py --interactive
```

## Supported File Types

| Format | Loader | OCR Support | Metadata |
|--------|--------|-------------|----------|
| PDF | PyPDFLoader | ✅ | Page count, creation date |
| DOCX/DOC | UnstructuredWordDocumentLoader | ✅ | Author, title, word count |
| TXT | TextLoader | ❌ | Encoding, size |
| JSON | JSONLoader | ❌ | Schema validation |
| CSV | CSVLoader | ❌ | Column info, row count |
| HTML | UnstructuredHTMLLoader | ✅ | Title, links, structure |
| Markdown | UnstructuredMarkdownLoader | ❌ | Headers, links |
| XLSX/XLS | UnstructuredExcelLoader | ❌ | Sheet info, formulas |
| PPTX/PPT | UnstructuredPowerPointLoader | ✅ | Slide count, notes |
| Images (PNG/JPG) | UnstructuredImageLoader | ✅ | EXIF data, dimensions |
| Jupyter Notebooks | NotebookLoader | ❌ | Cell metadata, outputs |

## Processing Strategies

### 1. Recursive Character Splitting (Default)
- **Best for**: General documents, mixed content
- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters (20%)
- **Separators**: `["\n\n", "\n", " ", ""]`

```python
config = ProcessingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    chunking_strategy="recursive"
)
```

### 2. Token-Based Splitting
- **Best for**: LLM optimization, precise token control
- **Uses**: tiktoken for accurate token counting
- **Models**: GPT-3.5, GPT-4, etc.

```bash
python scripts/ingest.py --input document.pdf --strategy token --chunk-size 512
```

### 3. Character Splitting
- **Best for**: Consistent chunk sizes, simple text
- **Fixed**: Exact character boundaries
- **Fast**: Minimal processing overhead

```bash
python scripts/ingest.py --input document.txt --strategy character --chunk-size 1500
```

## Advanced Usage

### Custom Processing Configuration

```python
from scripts.ingest import ProcessingConfig, LangChainDocumentProcessor

# Advanced configuration
config = ProcessingConfig(
    chunk_size=1500,
    chunk_overlap=300,
    chunking_strategy="recursive",
    max_workers=8,
    enable_ocr=True,
    extract_metadata=True,
    include_images=True,
    output_format="jsonl"
)

processor = LangChainDocumentProcessor(config)
chunks = processor.process_file(Path("document.pdf"), Path("output/"))
```

### Batch Processing with Filtering

```python
# Process specific file types
supported_extensions = {'.pdf', '.docx', '.txt', '.json', '.csv'}
files_to_process = [
    f for f in input_dir.rglob('*')
    if f.suffix.lower() in supported_extensions
]

# Parallel processing
all_chunks = processor.process_directory(input_dir, output_dir)
```

### Output Formats

#### JSON Format (Default)
```json
[
  {
    "chunk_id": "abc123_0001",
    "content": "This is the document content...",
    "chunk_index": 0,
    "tokens": 45,
    "metadata": {
      "source": "/path/to/document.pdf",
      "page": 1,
      "loader": "PyPDFLoader"
    },
    "source_metadata": {
      "source_path": "/path/to/document.pdf",
      "file_type": "pdf",
      "file_size": 1048576,
      "chunk_count": 25,
      "processing_time": 2.34
    }
  }
]
```

#### JSONL Format (Streaming)
```bash
python scripts/ingest.py --input document.pdf --output-format jsonl
```

Each line is a complete JSON object - optimal for streaming and large datasets.

### Error Handling and Recovery

The system includes comprehensive error handling:
- **Fallback loaders** for unsupported formats
- **Graceful degradation** when OCR fails  
- **Processing summaries** with error logs
- **Partial success** tracking for batch operations

### Performance Optimization

#### Memory Management
```python
# Process large directories in batches
config.max_workers = min(8, os.cpu_count())  # Optimize for your system
```

#### Storage Efficiency
```bash
# Use JSONL for large datasets (streaming friendly)
python scripts/ingest.py --input-dir large_docs/ --output-format jsonl --max-workers 12
```

## Integration with RAG

### 1. Generate Chunks for Vector Index
```bash
# Process documents for RAG pipeline
python scripts/ingest.py --input-dir documents/ --output-dir rag_chunks/ \
  --chunk-size 1000 --chunk-overlap 200 --output-format json
```

### 2. Load into RAG System
```python
# Integration with existing mini_rag.py
import json
from pathlib import Path

def load_processed_chunks(chunks_dir: Path):
    """Load pre-processed chunks into RAG system."""
    all_chunks = []
    
    for chunk_file in chunks_dir.glob("*_chunks.json"):
        with open(chunk_file) as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    
    return all_chunks

# Use in mini_rag.py
chunks = load_processed_chunks(Path("rag_chunks/"))
# Build vector index from chunks...
```

### 3. Metadata-Enhanced Retrieval
```python
# Filter chunks by metadata
pdf_chunks = [
    chunk for chunk in chunks 
    if chunk['source_metadata']['file_type'] == 'pdf'
]

recent_chunks = [
    chunk for chunk in chunks
    if chunk['source_metadata']['processing_time'] < 5.0  # Fast processing = simple docs
]
```

## Example Workflows

### Workflow 1: Research Paper Processing
```bash
# 1. Process academic PDFs with OCR
python scripts/ingest.py --input-dir research_papers/ --chunk-size 1500 --enable-ocr

# 2. Generate embeddings (using existing system)
cd apps/rag
python mini_rag.py --build-index --data ../../processed/

# 3. Query the knowledge base
python mini_rag.py --query "What are the key findings on neural networks?"
```

### Workflow 2: Enterprise Document Management
```bash
# 1. Batch process company documents
python scripts/ingest.py --input-dir company_docs/ --output-dir knowledge_base/ \
  --max-workers 16 --chunk-size 1000 --output-format jsonl --verbose

# 2. Monitor processing
tail -f processing.log

# 3. Validate results  
python scripts/ingest.py --input-dir knowledge_base/ --validate-chunks
```

### Workflow 3: Multi-Format Legal Documents
```bash
# 1. Process mixed legal documents (PDF, DOCX, TXT)
python scripts/ingest.py --input-dir legal_docs/ --strategy recursive \
  --chunk-size 1200 --chunk-overlap 300 --extract-metadata

# 2. Generate case law index
python apps/rag/mini_rag.py --build-index --data ../../chunks/ --domain legal

# 3. Legal research queries
python apps/rag/mini_rag.py --query "contract termination clauses" --top-k 10
```

## Monitoring and Analytics

### Processing Summary
Every batch operation generates a summary:
```json
{
  "processed_files": 150,
  "total_chunks": 3420,
  "processing_time": "2024-04-20T10:30:00",
  "average_chunks_per_file": 22.8,
  "file_types": {
    "pdf": 89,
    "docx": 45, 
    "txt": 16
  },
  "errors": [],
  "config": {...}
}
```

### Chunk Quality Metrics
- **Token distribution** per chunk
- **Content overlap** analysis  
- **Metadata completeness** scoring
- **Processing speed** by file type

## Troubleshooting

### Common Issues

#### 1. LangChain Import Errors
```bash
pip install langchain langchain-community langchain-text-splitters
```

#### 2. OCR Dependencies Missing
```bash
# Install tesseract
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Install Python packages  
pip install pytesseract pillow
```

#### 3. Memory Issues with Large Files
```python
# Reduce workers and chunk size
config.max_workers = 2
config.chunk_size = 500
```

#### 4. Encoding Problems
```python
# The system auto-handles encoding detection
# For manual override:
loader = TextLoader(file_path, encoding='utf-8')
```

### Performance Tips

1. **CPU Optimization**: Set `max_workers` to `cpu_count() - 1`
2. **Memory Management**: Use JSONL for large datasets  
3. **Storage**: Enable compression for output files
4. **OCR Speed**: Disable OCR for text-only documents

## Next Steps

1. **Integration**: Connect processed chunks to vector database
2. **Enhancement**: Add custom metadata extractors
3. **Monitoring**: Implement real-time processing dashboards
4. **Scaling**: Deploy on cloud infrastructure for large-scale processing

---

**📚 Document Processing Complete!** Your files are now ready for RAG applications with rich metadata and optimal chunking for LLM consumption.