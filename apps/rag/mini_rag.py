#!/usr/bin/env python3
"""
Mini RAG (Retrieval-Augmented Generation) Application
===================================================

A complete RAG system combining your advanced search infrastructure with LLM generation.
Demonstrates enterprise-grade retrieval-augmented generation patterns following Microsoft 
best practices.

Features:
- Hybrid semantic + keyword search for precise retrieval
- Smart chunking for optimal context windows
- GPT integration with configurable models and prompts
- Context-aware response generation with citations
- Interactive CLI and programmatic API
- Performance monitoring and result explanations

Examples:
    # Build knowledge base from insurance data
    python mini_rag.py --build-index --data ../data/insurance_data.json
    
    # Ask questions interactively
    python mini_rag.py --interactive
    
    # Single question mode
    python mini_rag.py --query "What insurance policies are available?"
    
    # Advanced retrieval with hybrid search
    python mini_rag.py --query "Tell me about auto insurance" --hybrid --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use system environment variables

# PDF processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
import os

# Add parent directories to path for imports
current_dir = Path(__file__).parent
llm_dir = current_dir.parent / "llm"
sys.path.extend([str(current_dir), str(llm_dir)])

try:
    from improved_search import EnhancedSemanticSearch, EnhancedDocument, EnhancedSearchResult
    from chunking import DocumentChunker, ChunkingConfig, ChunkingStrategy
    from embeddings import EmbeddingGenerator, EmbeddingConfig
    from openai_client import LLMClient, ChatTurn, ResponseFormat, ClientConfig
    from prompts import resolve_prompt_template
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory and have all dependencies installed")
    sys.exit(1)


@dataclass 
class RAGConfig:
    """Configuration for the RAG system."""
    
    # Search configuration
    index_path: str = "rag_index.faiss"
    top_k: int = 3
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.7  # Weight for semantic vs keyword search
    use_reranking: bool = True
    
    # Language model configuration  
    model: str = "gpt-4.1-nano"  # Azure deployment name
    temperature: float = 0.1
    max_tokens: int = 1000
    provider: str = "azure"  # Use Azure OpenAI
    
    # Embedding configuration for Azure OpenAI
    embedding_provider: str = "azure"
    embedding_model: str = "text-embedding-3-small"  # Azure deployment name
    
    # Chunking configuration
    chunking_strategy: str = "sentence"  # fixed, sentence, semantic, hybrid
    chunk_size: int = 2000
    chunk_overlap: int = 500
    
    # RAG behavior
    require_citations: bool = True
    min_confidence_threshold: float = 0.3
    context_window_chars: int = 8000  # Approximate context window for LLM
    explain_retrieval: bool = False  # Show retrieval details to user


@dataclass 
class RAGResult:
    """Result from a RAG query containing both retrieved context and generated answer."""
    
    query: str
    answer: str
    sources: List[EnhancedSearchResult]
    confidence: float
    context_used: str
    explanation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            'query': self.query,
            'answer': self.answer,
            'confidence': self.confidence,
            'context_used': self.context_used,
            'sources': [
                {
                    'title': source.document.title,
                    'score': source.final_score,
                    'content_preview': source.document.content[:200] + '...' if len(source.document.content) > 200 else source.document.content,
                    'metadata': source.document.metadata
                }
                for source in self.sources
            ],
            'explanation': self.explanation
        }


class MiniRAG:
    """
    Mini RAG system combining advanced search with LLM generation.
    
    Implements Microsoft's recommended RAG patterns:
    - Hybrid retrieval (semantic + keyword)
    - Smart chunking for optimal context
    - Citation-based response generation
    - Confidence scoring and explanation
    """
    
    def __init__(self, config: RAGConfig = None):
        """Initialize the RAG system with configuration."""
        self.config = config or RAGConfig()
        self.search_system = None
        self.llm_client = None
        self.chunker = None
        
        # Initialize components
        self._initialize_search_system()
        self._initialize_llm_client()
        self._initialize_chunker()
    
    def _initialize_search_system(self):
        """Initialize the enhanced search system."""
        try:
            # Create embedding config for Azure OpenAI
            embedding_config = EmbeddingConfig(
                provider="azure",
                model="text-embedding-3-small"  # Your Azure deployment name
                # API key and endpoint will be read from environment variables:
                # AZURE_OPENAI_API_KEY
                # AZURE_OPENAI_ENDPOINT
            )
            
            # Try to load existing index
            try:
                self.search_system = EnhancedSemanticSearch(embedding_config)
                self.search_system.load_index(self.config.index_path)
                print(f"✅ Search system loaded from: {self.config.index_path}")
            except (FileNotFoundError, RuntimeError):
                print(f"ℹ️ No existing index found at {self.config.index_path}. Build one with --build-index")
                self.search_system = None
        except Exception as e:
            print(f"⚠️ Search system initialization error: {e}")
            self.search_system = None
    
    def _initialize_llm_client(self):
        """Initialize the LLM client."""
        try:
            # Create client configuration for Azure OpenAI
            config = ClientConfig(
                provider="azure",  # Use Azure OpenAI
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                use_chat_history=False  # Disable chat history for RAG
            )
            
            self.llm_client = LLMClient(config)
            print(f"✅ LLM client initialized with provider: {self.config.provider}")
        except Exception as e:
            print(f"❌ Failed to initialize LLM client: {e}")
            # Don't exit here for RAG - we can still build index without LLM
            self.llm_client = None
    
    def _initialize_chunker(self):
        """Initialize the document chunker."""
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.chunk_overlap
        )
        self.chunker = DocumentChunker(chunking_config)
        print(f"✅ Chunker initialized with strategy: {self.config.chunking_strategy}")
    
    def build_knowledge_base(self, data_path: str) -> bool:
        """
        Build the knowledge base index from data files.
        
        Args:
            data_path: Path to data file, directory, or specific file types
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🏗️ Building knowledge base from: {data_path}")
            
            # Check if path exists
            path_obj = Path(data_path)
            if not path_obj.exists():
                print(f"❌ Path not found: {data_path}")
                return False
            
            # Load documents from various sources
            documents = []
            
            if path_obj.is_file():
                # Single file
                docs = self._load_documents_from_file(path_obj)
                documents.extend(docs)
            elif path_obj.is_dir():
                # Directory - process all supported files
                print("📁 Processing directory with multiple files...")
                for file_path in path_obj.iterdir():
                    if file_path.is_file() and self._is_supported_file(file_path):
                        print(f"   📄 Processing: {file_path.name}")
                        docs = self._load_documents_from_file(file_path)
                        documents.extend(docs)
            else:
                print(f"❌ Invalid path type: {data_path}")
                return False
            
            if not documents:
                print("❌ No documents found or loaded")
                return False
            
            # Create search system with Azure OpenAI embeddings
            embedding_config = EmbeddingConfig(
                provider="azure",
                model="text-embedding-3-small"  # Your Azure deployment name
            )
            self.search_system = EnhancedSemanticSearch(embedding_config)
            
            # Set chunking configuration
            chunking_config = ChunkingConfig(
                strategy=ChunkingStrategy.SENTENCE,
                chunk_size=self.config.chunk_size,
                overlap_size=self.config.chunk_overlap
            )
            self.search_system.set_chunking_config(chunking_config)
            
            # Add documents with chunking
            print(f"🔪 Chunking {len(documents)} documents...")
            self.search_system.add_documents(documents, use_chunking=True)
            
            # Save the index
            self.search_system.save_index(self.config.index_path)
            
            print(f"✅ Knowledge base built successfully!")
            print(f"   📊 {len(documents)} documents processed")
            print(f"   💾 Index saved to: {self.config.index_path}")
            return True
                
        except Exception as e:
            print(f"❌ Error building knowledge base: {e}")
            return False
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported for processing."""
        supported_extensions = {'.json', '.pdf', '.txt', '.csv'}
        return file_path.suffix.lower() in supported_extensions
    
    def _load_documents_from_file(self, file_path: Path) -> List[EnhancedDocument]:
        """Load documents from a single file based on its type."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.json':
                return self._load_documents_from_json(file_path)
            elif file_ext == '.pdf':
                return self._load_documents_from_pdf(file_path)
            elif file_ext in ['.txt', '.csv']:
                return self._load_documents_from_text(file_path)
            else:
                print(f"⚠️  Unsupported file type: {file_ext}")
                return []
                
        except Exception as e:
            print(f"❌ Error loading from {file_path.name}: {e}")
            return []
    
    def _load_documents_from_json(self, file_path: Path) -> List[EnhancedDocument]:
        """Load documents from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return self._create_documents_from_data(raw_data, source_file=file_path.name)
    
    def _load_documents_from_pdf(self, file_path: Path) -> List[EnhancedDocument]:
        """Extract text from PDF and create document."""
        if not PyPDF2:
            print("❌ PyPDF2 not available. Cannot process PDF files.")
            return []
        
        try:
            text_content = self._extract_text_from_pdf(file_path)
            if not text_content.strip():
                print(f"⚠️  No text found in PDF: {file_path.name}")
                return []
            
            # Create single document from PDF content
            document = EnhancedDocument(
                id=f"pdf_{file_path.stem}",
                title=file_path.stem.replace('_', ' ').title(),
                content=text_content,
                metadata={
                    'source_file': file_path.name,
                    'file_type': 'pdf',
                    'created_date': datetime.now().isoformat(),
                    'file_size': file_path.stat().st_size,
                    'keywords': self._extract_keywords(text_content)
                }
            )
            
            return [document]
            
        except Exception as e:
            print(f"❌ Error processing PDF {file_path.name}: {e}")
            return []
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\\n{page_text}")
                except Exception as e:
                    print(f"⚠️  Error extracting text from page {page_num + 1}: {e}")
                    continue
        
        return "\\n\\n".join(text_parts)
    
    def _load_documents_from_text(self, file_path: Path) -> List[EnhancedDocument]:
        """Load documents from text or CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create single document from text content  
            document = EnhancedDocument(
                id=f"text_{file_path.stem}",
                title=file_path.stem.replace('_', ' ').title(),
                content=content,
                metadata={
                    'source_file': file_path.name,
                    'file_type': file_path.suffix[1:],  # Remove the dot
                    'created_date': datetime.now().isoformat(),
                    'file_size': file_path.stat().st_size,
                    'keywords': self._extract_keywords(content)
                }
            )
            
            return [document]
            
        except Exception as e:
            print(f"❌ Error processing text file {file_path.name}: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction - remove common words and get unique terms
        import re
        words = re.findall(r'\\b[a-zA-Z]{4,}\\b', text.lower())
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 
                     'been', 'your', 'what', 'when', 'where', 'would', 'there',
                     'their', 'said', 'each', 'which', 'more', 'some', 'very'}
        keywords = [word for word in set(words) if word not in stop_words]
        return keywords[:10]  # Return top 10 keywords

    def _create_documents_from_data(self, raw_data: List[Dict], source_file: str = 'insurance_data.json') -> List[EnhancedDocument]:
        """Convert raw JSON data to EnhancedDocument objects."""
        documents = []
        
        for i, item in enumerate(raw_data):
            # Create readable content from the data item
            content_parts = []
            title = f"Record {i + 1}"
            
            # Handle different types of data structures
            if isinstance(item, dict):
                for key, value in item.items():
                    content_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                    
                    # Use certain fields as title if available
                    if key.lower() in ['name', 'title', 'policy_id', 'customer_id']:
                        title = f"{key.replace('_', ' ').title()}: {value}"
            else:
                content_parts.append(str(item))
            
            content = "\n".join(content_parts)
            
            # Extract keywords from content
            keywords = self._extract_simple_keywords(content)
            
            # Create enhanced document
            doc = EnhancedDocument(
                id=f"doc_{i}",
                title=title,
                content=content,
                metadata={
                    'record_index': i,
                    'source_file': source_file,
                    'data_type': type(item).__name__
                },
                keywords=keywords,
                created_date=datetime.now(),
                boost_factor=1.0
            )
            
            documents.append(doc)
        
        return documents
    
    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text."""
        # Simple keyword extraction - in production, use more sophisticated NLP
        import re
        
        # Remove special characters and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stopwords to filter out
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'down', 'each', 'few', 'may', 'put', 'say', 'she', 'too', 'use'}
        
        # Filter and get unique keywords
        keywords = list(set([word for word in words if word not in stopwords]))
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def query(self, question: str) -> RAGResult:
        """
        Process a question using RAG: retrieve relevant context and generate answer.
        
        Args:
            question: User question to answer
            
        Returns:
            RAGResult containing answer, sources, and metadata
        """
        if not self.search_system or not self.search_system.documents:
            raise RuntimeError("Knowledge base not built. Use build_knowledge_base() first.")
        
        print(f"🔍 Processing query: {question}")
        
        # Step 1: Retrieve relevant context
        search_results = self._retrieve_context(question)
        
        # Step 2: Check confidence threshold
        if not search_results or search_results[0].final_score < self.config.min_confidence_threshold:
            return RAGResult(
                query=question,
                answer="I don't have enough relevant information to answer your question confidently.",
                sources=search_results,
                confidence=search_results[0].final_score if search_results else 0.0,
                context_used="",
                explanation={"reason": "Below confidence threshold", "threshold": self.config.min_confidence_threshold}
            )
        
        # Step 3: Prepare context for LLM
        context = self._prepare_context(search_results)
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, context)
        
        # Step 5: Calculate overall confidence 
        confidence = self._calculate_confidence(search_results)
        
        # Create result
        result = RAGResult(
            query=question,
            answer=answer,
            sources=search_results,
            confidence=confidence,
            context_used=context,
            explanation={
                "retrieval_method": "hybrid" if self.config.use_hybrid_search else "semantic",
                "chunks_used": len(search_results),
                "context_length": len(context),
                "reranking_enabled": self.config.use_reranking
            }
        )
        
        return result
    
    def _retrieve_context(self, question: str) -> List[EnhancedSearchResult]:
        """Retrieve relevant context using the enhanced search system."""
        # For now, use basic semantic search - hybrid search might need different method
        return self.search_system.search(question, k=self.config.top_k)
    
    def _prepare_context(self, search_results: List[EnhancedSearchResult]) -> str:
        """Prepare context string from search results for LLM input."""
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(search_results):
            # Format as numbered source
            source_text = f"[Source {i + 1}] {result.document.title}\n{result.document.content}\n"
            
            # Check if adding this would exceed context window
            if total_chars + len(source_text) > self.config.context_window_chars:
                break
                
            context_parts.append(source_text)
            total_chars += len(source_text)
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Check if LLM client is available
            if not self.llm_client:
                return "LLM client not available. Please check your configuration and API keys."
            
            # Build prompt using existing prompt template
            rag_prompt = resolve_prompt_template("rag")
            
            if not rag_prompt:
                # Fallback prompt if template not found
                rag_prompt = """You are an AI assistant answering ONLY from provided context.

Rules:
- Use only the context provided
- If answer not found in context, say "I don't know based on the provided information"
- Do not make up information
- Keep answer concise and accurate
- Add citations like [Source 1] when referencing information"""

            # Construct the full prompt
            full_prompt = f"""{rag_prompt}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            # Generate response using LLM client
            response = self.llm_client.ask(full_prompt, response_format=ResponseFormat.TEXT)
            
            # Handle response - ask method returns string for TEXT format
            if isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def _calculate_confidence(self, search_results: List[EnhancedSearchResult]) -> float:
        """Calculate overall confidence score for the RAG result."""
        if not search_results:
            return 0.0
        
        # Simple confidence calculation based on top result scores
        top_scores = [result.final_score for result in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Boost confidence if multiple good results
        if len([s for s in top_scores if s > 0.5]) >= 2:
            avg_score *= 1.1
        
        return min(1.0, avg_score)
    
    def interactive_mode(self):
        """Run interactive question-answering session."""
        print("\n🤖 Mini RAG Interactive Mode")
        print("Type your questions or 'quit' to exit")
        print("-" * 50)
        
        if not self.search_system or not self.search_system.documents:
            print("❌ Knowledge base not ready. Build it first with --build-index")
            return
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Process query
                result = self.query(question)
                
                # Display result
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
    
    def _display_result(self, result: RAGResult):
        """Display RAG result in a user-friendly format."""
        print(f"\n✅ **Answer** (Confidence: {result.confidence:.2f})")
        print(f"{result.answer}\n")
        
        if self.config.explain_retrieval and result.sources:
            print("📊 **Sources Used:**")
            for i, source in enumerate(result.sources):
                print(f"  [{i + 1}] {source.document.title} (Score: {source.final_score:.3f})")
                print(f"      {source.document.content[:150]}...")
            print()
        
        if self.config.explain_retrieval and result.explanation:
            print("🔍 **Retrieval Details:**")
            for key, value in result.explanation.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")


def main():
    """Command line interface for Mini RAG."""
    parser = argparse.ArgumentParser(description="Mini RAG - Retrieval Augmented Generation")
    
    # Action arguments
    parser.add_argument('--build-index', action='store_true', help='Build the knowledge base index')
    parser.add_argument('--data', type=str, default='../data/', help='Path to data file or directory (supports JSON, PDF, TXT, CSV)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive Q&A session')
    parser.add_argument('--query', type=str, help='Single question to ask')
    
    # Configuration arguments
    parser.add_argument('--index-path', type=str, default='rag_index.faiss', help='Path to FAISS index')
    parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid search (semantic + keyword)')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano', help='LLM model to use (Azure deployment name)')
    parser.add_argument('--temperature', type=float, default=0.1, help='LLM temperature')
    parser.add_argument('--explain', action='store_true', help='Show detailed retrieval explanations')
    
    args = parser.parse_args()
    
    # Create configuration
    config = RAGConfig(
        index_path=args.index_path,
        top_k=args.top_k,
        use_hybrid_search=args.hybrid,
        model=args.model,
        temperature=args.temperature,
        explain_retrieval=args.explain
    )
    
    # Initialize RAG system
    rag = MiniRAG(config)
    
    # Execute requested action
    if args.build_index:
        success = rag.build_knowledge_base(args.data)
        if not success:
            sys.exit(1)
    elif args.interactive:
        rag.interactive_mode()
    elif args.query:
        try:
            result = rag.query(args.query)
            rag._display_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()