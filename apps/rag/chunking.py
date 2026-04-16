#!/usr/bin/env python3
"""
Advanced Document Chunking System for Enterprise AI Assistant
===============================================================

Implements multiple chunking strategies based on Microsoft Azure Search best practices:
- Fixed-size chunks with overlap (recommended: 512 tokens, 25% overlap)
- Sentence-aware chunking 
- Semantic chunking
- Custom hybrid approaches

Based on: https://learn.microsoft.com/azure/search/vector-search-how-to-chunk-documents

Examples:
    # Fixed-size chunking (recommended)
    python chunking.py --input document.txt --strategy fixed --chunk-size 2000 --overlap 500
    
    # Sentence-aware chunking
    python chunking.py --input document.txt --strategy sentence --max-chunk-size 1500
    
    # Interactive chunking exploration
    python chunking.py --interactive
"""

import argparse
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sys

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Import NLTK for sentence tokenization (optional)
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📦 Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    nltk = None
    sent_tokenize = None


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED = "fixed"           # Fixed-size chunks with overlap
    SENTENCE = "sentence"     # Sentence-boundary aware chunks 
    SEMANTIC = "semantic"     # Semantic/paragraph-based chunks
    HYBRID = "hybrid"         # Combination of strategies


@dataclass
class Chunk:
    """A text chunk with metadata."""
    
    id: str
    text: str
    char_count: int
    token_count: Optional[int]
    source_document: str
    chunk_index: int
    start_position: int
    end_position: int
    overlap_with_previous: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create Chunk from dictionary."""
        return cls(**data)


@dataclass 
class ChunkingConfig:
    """Configuration for chunking strategies."""
    
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED
    chunk_size: int = 2000              # Characters (Microsoft recommends ~2000)
    overlap_size: int = 500             # Characters (Microsoft recommends 25% = ~500)
    max_chunk_size: int = 4000          # Maximum chunk size
    min_chunk_size: int = 100           # Minimum viable chunk size
    preserve_sentences: bool = True     # Don't break sentences when possible
    preserve_paragraphs: bool = False   # Try to keep paragraphs intact
    tokenizer_model: str = "gpt-4"      # For token counting
    
    def get_overlap_ratio(self) -> float:
        """Get overlap as percentage of chunk size."""
        return self.overlap_size / self.chunk_size if self.chunk_size > 0 else 0


class DocumentChunker:
    """Advanced document chunking system with multiple strategies."""
    
    def __init__(self, config: ChunkingConfig = None):
        """Initialize with chunking configuration."""
        self.config = config or ChunkingConfig()
        self.tokenizer = None
        
        # Initialize tokenizer if available
        if tiktoken:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.config.tokenizer_model)
            except Exception as e:
                print(f"⚠️  Could not load tokenizer for {self.config.tokenizer_model}: {e}")
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default GPT-4 encoding
                except Exception:
                    print("⚠️  Could not load default tokenizer. Token counts unavailable.")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not self.tokenizer:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return len(text) // 4
    
    def chunk_document(self, text: str, document_name: str = "document") -> List[Chunk]:
        """Chunk a document using the configured strategy."""
        
        if self.config.strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed_size(text, document_name)
        elif self.config.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentence_aware(text, document_name)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, document_name)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            return self._chunk_hybrid(text, document_name)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def _chunk_fixed_size(self, text: str, document_name: str) -> List[Chunk]:
        """
        Fixed-size chunking with overlap (Microsoft recommended approach).
        
        Creates chunks of approximately chunk_size characters with overlap_size overlap.
        Tries to respect sentence boundaries when preserve_sentences=True.
        """
        chunks = []
        
        if len(text) <= self.config.chunk_size:
            # Document is small enough, return as single chunk
            chunk = Chunk(
                id=f"{document_name}_chunk_0",
                text=text.strip(),
                char_count=len(text),
                token_count=self.count_tokens(text),
                source_document=document_name,
                chunk_index=0,
                start_position=0,
                end_position=len(text),
                metadata={"strategy": "fixed", "is_single_chunk": True}
            )
            return [chunk]
        
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Calculate chunk end position
            chunk_end = min(current_pos + self.config.chunk_size, len(text))
            
            # If preserving sentences, try to end at sentence boundary
            if self.config.preserve_sentences and chunk_end < len(text):
                chunk_end = self._find_sentence_boundary(text, chunk_end, direction="backward")
                
                # If we couldn't find a good sentence boundary, use original position
                if chunk_end <= current_pos:
                    chunk_end = min(current_pos + self.config.chunk_size, len(text))
            
            chunk_text = text[current_pos:chunk_end].strip()
            
            if chunk_text:  # Only create non-empty chunks
                # Calculate overlap with previous chunk
                overlap = 0
                if chunk_index > 0:
                    overlap = max(0, self.config.overlap_size - (current_pos - chunks[-1].end_position + len(chunks[-1].text)))
                
                chunk = Chunk(
                    id=f"{document_name}_chunk_{chunk_index}",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=self.count_tokens(chunk_text),
                    source_document=document_name,
                    chunk_index=chunk_index,
                    start_position=current_pos,
                    end_position=chunk_end,
                    overlap_with_previous=overlap,
                    metadata={
                        "strategy": "fixed",
                        "target_size": self.config.chunk_size,
                        "overlap_size": self.config.overlap_size
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next position with overlap
            if chunk_end >= len(text):
                break
                
            # Calculate next starting position with overlap
            next_pos = chunk_end - self.config.overlap_size
            
            # Ensure we make progress
            if next_pos <= current_pos:
                next_pos = current_pos + 1
                
            current_pos = max(0, next_pos)
        
        return chunks
    
    def _chunk_sentence_aware(self, text: str, document_name: str) -> List[Chunk]:
        """
        Sentence-aware chunking that respects sentence boundaries.
        
        Creates chunks by combining sentences until max_chunk_size is reached.
        """
        if not sent_tokenize:
            print("⚠️  NLTK not available. Falling back to fixed chunking.")
            return self._chunk_fixed_size(text, document_name)
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_index = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed max chunk size
            if (current_length + sentence_length > self.config.max_chunk_size and 
                current_chunk_sentences):
                
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunk_start = char_position - current_length
                
                chunk = Chunk(
                    id=f"{document_name}_chunk_{chunk_index}",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=self.count_tokens(chunk_text),
                    source_document=document_name,
                    chunk_index=chunk_index,
                    start_position=chunk_start,
                    end_position=char_position,
                    metadata={
                        "strategy": "sentence",
                        "sentence_count": len(current_chunk_sentences)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            char_position += sentence_length + 1
        
        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunk_start = char_position - current_length
            
            chunk = Chunk(
                id=f"{document_name}_chunk_{chunk_index}",
                text=chunk_text,
                char_count=len(chunk_text),
                token_count=self.count_tokens(chunk_text),
                source_document=document_name,
                chunk_index=chunk_index,
                start_position=chunk_start,
                end_position=char_position,
                metadata={
                    "strategy": "sentence", 
                    "sentence_count": len(current_chunk_sentences),
                    "is_final_chunk": True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantic(self, text: str, document_name: str) -> List[Chunk]:
        """
        Semantic chunking based on paragraph boundaries.
        
        Splits on double newlines (paragraphs) and combines them into 
        appropriately sized chunks.
        """
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk_paras = []
        current_length = 0
        chunk_index = 0
        char_position = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            
            # Check if adding this paragraph would exceed max chunk size
            if (current_length + para_length > self.config.max_chunk_size and 
                current_chunk_paras):
                
                # Create chunk from current paragraphs
                chunk_text = "\n\n".join(current_chunk_paras).strip()
                chunk_start = char_position - current_length
                
                chunk = Chunk(
                    id=f"{document_name}_chunk_{chunk_index}",
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=self.count_tokens(chunk_text),
                    source_document=document_name,
                    chunk_index=chunk_index,
                    start_position=chunk_start,
                    end_position=char_position,
                    metadata={
                        "strategy": "semantic",
                        "paragraph_count": len(current_chunk_paras)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk
                current_chunk_paras = [paragraph]
                current_length = para_length
            else:
                # Add paragraph to current chunk
                current_chunk_paras.append(paragraph)
                current_length += para_length + 2  # +2 for \n\n
            
            char_position += para_length + 2
        
        # Handle remaining paragraphs
        if current_chunk_paras:
            chunk_text = "\n\n".join(current_chunk_paras).strip()
            chunk_start = char_position - current_length
            
            chunk = Chunk(
                id=f"{document_name}_chunk_{chunk_index}",
                text=chunk_text,
                char_count=len(chunk_text),
                token_count=self.count_tokens(chunk_text),
                source_document=document_name,
                chunk_index=chunk_index,
                start_position=chunk_start,
                end_position=char_position,
                metadata={
                    "strategy": "semantic",
                    "paragraph_count": len(current_chunk_paras),
                    "is_final_chunk": True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid(self, text: str, document_name: str) -> List[Chunk]:
        """
        Hybrid chunking that combines semantic and fixed-size approaches.
        
        First tries semantic chunking, then applies fixed-size splitting 
        to any chunks that are still too large.
        """
        # Start with semantic chunking
        semantic_chunks = self._chunk_semantic(text, document_name)
        
        final_chunks = []
        chunk_index = 0
        
        for semantic_chunk in semantic_chunks:
            # If semantic chunk is within size limits, keep it
            if semantic_chunk.char_count <= self.config.chunk_size:
                semantic_chunk.id = f"{document_name}_chunk_{chunk_index}"
                semantic_chunk.chunk_index = chunk_index
                semantic_chunk.metadata["strategy"] = "hybrid_semantic"
                final_chunks.append(semantic_chunk)
                chunk_index += 1
            else:
                # Apply fixed-size chunking to oversized semantic chunk  
                sub_chunks = self._chunk_fixed_size(semantic_chunk.text, f"{document_name}_semantic_{semantic_chunk.chunk_index}")
                
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{document_name}_chunk_{chunk_index}"
                    sub_chunk.chunk_index = chunk_index
                    sub_chunk.source_document = document_name
                    sub_chunk.metadata["strategy"] = "hybrid_fixed"
                    sub_chunk.metadata["derived_from_semantic"] = True
                    final_chunks.append(sub_chunk)
                    chunk_index += 1
        
        return final_chunks
    
    def _find_sentence_boundary(self, text: str, position: int, direction: str = "backward") -> int:
        """Find the nearest sentence boundary from a given position."""
        sentence_endings = ".!?"
        max_search = 200  # Don't search too far
        
        if direction == "backward":
            # Search backward for sentence ending
            for i in range(min(max_search, position)):
                pos = position - i
                if pos > 0 and text[pos-1] in sentence_endings:
                    # Check if next char is space or end
                    if pos >= len(text) or text[pos].isspace():
                        return pos
            return position
        else:
            # Search forward for sentence ending
            for i in range(min(max_search, len(text) - position)):
                pos = position + i
                if pos < len(text) - 1 and text[pos] in sentence_endings:
                    # Check if next char is space
                    if pos + 1 >= len(text) or text[pos + 1].isspace():
                        return pos + 1
            return position


def analyze_chunks(chunks: List[Chunk]) -> Dict[str, Any]:
    """Analyze chunk statistics for optimization."""
    if not chunks:
        return {}
    
    char_counts = [chunk.char_count for chunk in chunks]
    token_counts = [chunk.token_count for chunk in chunks if chunk.token_count]
    overlaps = [chunk.overlap_with_previous for chunk in chunks]
    
    analysis = {
        "total_chunks": len(chunks),
        "total_characters": sum(char_counts),
        "avg_chunk_size": sum(char_counts) / len(char_counts),
        "min_chunk_size": min(char_counts),
        "max_chunk_size": max(char_counts),
        "char_size_std": math.sqrt(sum((x - sum(char_counts) / len(char_counts))**2 for x in char_counts) / len(char_counts)),
        "strategies_used": list(set(chunk.metadata.get("strategy", "unknown") for chunk in chunks))
    }
    
    if token_counts:
        analysis.update({
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chars_per_token": sum(char_counts) / sum(token_counts) if sum(token_counts) > 0 else 4
        })
    
    if any(o > 0 for o in overlaps):
        analysis.update({
            "avg_overlap": sum(overlaps) / len(overlaps),
            "max_overlap": max(overlaps)
        })
    
    return analysis


def load_document(file_path: Path) -> str:
    """Load text document from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return ""


def save_chunks(chunks: List[Chunk], output_path: Path) -> None:
    """Save chunks to JSON file."""
    chunk_data = {
        "chunks": [chunk.to_dict() for chunk in chunks],
        "analysis": analyze_chunks(chunks)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")


def print_chunk_analysis(chunks: List[Chunk]) -> None:
    """Print detailed chunk analysis."""
    analysis = analyze_chunks(chunks)
    
    print("\n📊 Chunk Analysis:")
    print(f"  Total chunks: {analysis['total_chunks']}")
    print(f"  Total characters: {analysis['total_characters']:,}")
    print(f"  Average chunk size: {analysis['avg_chunk_size']:.0f} chars")
    print(f"  Size range: {analysis['min_chunk_size']} - {analysis['max_chunk_size']} chars")
    print(f"  Size std deviation: {analysis['char_size_std']:.0f}")
    print(f"  Strategies used: {', '.join(analysis['strategies_used'])}")
    
    if 'total_tokens' in analysis:
        print(f"  Total tokens: {analysis['total_tokens']:,}")
        print(f"  Average tokens per chunk: {analysis['avg_tokens_per_chunk']:.0f}")
        print(f"  Chars per token ratio: {analysis['chars_per_token']:.1f}")
    
    if 'avg_overlap' in analysis:
        print(f"  Average overlap: {analysis['avg_overlap']:.0f} chars")


def interactive_mode():
    """Interactive chunking exploration."""
    print("\n🧠 Document Chunking - Interactive Mode")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'help' for commands")
    print("💡 Type 'config' to show current configuration")
    print("-" * 50)
    
    # Default configuration
    config = ChunkingConfig()
    chunker = DocumentChunker(config)
    
    while True:
        try:
            command = input("\n📝 Command: ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif command == 'help':
                print("\n📋 Available commands:")
                print("  help          - Show this help") 
                print("  config        - Show current chunking configuration")
                print("  set <param>   - Set configuration parameter")
                print("  chunk <file>  - Chunk a document file")
                print("  test <text>   - Test chunking on provided text")
                print("  compare       - Compare strategies on sample text")
                print("  quit          - Exit interactive mode")
                continue
            
            elif command == 'config':
                print(f"\n⚙️ Current Configuration:")
                print(f"  Strategy: {config.strategy.value}")
                print(f"  Chunk size: {config.chunk_size} chars")
                print(f"  Overlap size: {config.overlap_size} chars ({config.get_overlap_ratio():.1%})")
                print(f"  Max chunk size: {config.max_chunk_size} chars")
                print(f"  Preserve sentences: {config.preserve_sentences}")
                print(f"  Preserve paragraphs: {config.preserve_paragraphs}")
                continue
                
            elif command.startswith('set '):
                # Simple parameter setting
                parts = command.split(' ', 2)
                if len(parts) >= 3:
                    param, value = parts[1], parts[2]
                    try:
                        if param == 'strategy':
                            config.strategy = ChunkingStrategy(value)
                            chunker = DocumentChunker(config)  # Recreate with new config
                        elif param in ['chunk_size', 'overlap_size', 'max_chunk_size']:
                            setattr(config, param, int(value))
                        elif param in ['preserve_sentences', 'preserve_paragraphs']:
                            setattr(config, param, value.lower() in ['true', '1', 'yes'])
                        print(f"✅ Set {param} = {value}")
                    except Exception as e:
                        print(f"❌ Error setting {param}: {e}")
                else:
                    print("❌ Usage: set <parameter> <value>")
                continue
                
            elif command.startswith('chunk '):
                file_path = Path(command[6:].strip())
                if file_path.exists():
                    print(f"📂 Loading {file_path}...")
                    text = load_document(file_path)
                    if text:
                        chunks = chunker.chunk_document(text, file_path.stem)
                        print_chunk_analysis(chunks)
                        
                        # Show first few chunks
                        print(f"\n📄 First 3 chunks:")
                        for chunk in chunks[:3]:
                            preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                            print(f"\n  Chunk {chunk.chunk_index} ({chunk.char_count} chars):")
                            print(f"    {preview}")
                else:
                    print(f"❌ File not found: {file_path}")
                continue
                
            elif command.startswith('test '):
                test_text = command[5:].strip()
                if test_text:
                    chunks = chunker.chunk_document(test_text, "test")
                    print_chunk_analysis(chunks)
                    
                    print(f"\n📄 Generated chunks:")
                    for chunk in chunks:
                        print(f"\n  Chunk {chunk.chunk_index}: {chunk.text}")
                else:
                    print("❌ Please provide text to test")
                continue
                
            elif command == 'compare':
                # Compare strategies on sample text
                sample_text = """
                Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding.

                Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data.

                Deep Learning uses artificial neural networks with multiple layers to model complex patterns in data. These networks are inspired by the human brain and can automatically learn hierarchical features.

                The future of AI holds tremendous promise across industries, from healthcare and finance to transportation and entertainment. However, it also raises important questions about ethics, privacy, and the impact on employment.
                """
                
                strategies = [ChunkingStrategy.FIXED, ChunkingStrategy.SENTENCE, ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID]
                
                print("\n🔄 Comparing chunking strategies on sample text:")
                for strategy in strategies:
                    test_config = ChunkingConfig(strategy=strategy, chunk_size=500, overlap_size=125)
                    test_chunker = DocumentChunker(test_config)
                    chunks = test_chunker.chunk_document(sample_text, "sample")
                    
                    print(f"\n  📊 {strategy.value.upper()} strategy:")
                    print(f"    Chunks created: {len(chunks)}")
                    if chunks:
                        avg_size = sum(c.char_count for c in chunks) / len(chunks)
                        print(f"    Average size: {avg_size:.0f} chars")
                        print(f"    Size range: {min(c.char_count for c in chunks)} - {max(c.char_count for c in chunks)}")
                continue
                
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced document chunking system with multiple strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed-size chunking (Microsoft recommended)
  python chunking.py --input document.txt --strategy fixed --chunk-size 2000 --overlap 500
  
  # Sentence-aware chunking  
  python chunking.py --input document.txt --strategy sentence --max-chunk-size 1500
  
  # Semantic (paragraph) chunking
  python chunking.py --input document.txt --strategy semantic
  
  # Hybrid approach
  python chunking.py --input document.txt --strategy hybrid
  
  # Interactive exploration
  python chunking.py --interactive
  
Recommended settings (Microsoft best practices):
  - Fixed chunking: 2000 chars, 500 overlap (25%)
  - For token-based: ~512 tokens, ~128 token overlap
"""
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--input', type=Path, 
                           help='Input document file to chunk')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Run interactive chunking mode')
    
    # Chunking strategy
    parser.add_argument('--strategy', type=str,
                       choices=[s.value for s in ChunkingStrategy],
                       default='fixed',
                       help='Chunking strategy (default: fixed)')
    
    # Size parameters
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='Target chunk size in characters (default: 2000)')
    parser.add_argument('--overlap', type=int, default=500,
                       help='Overlap size in characters (default: 500)')
    parser.add_argument('--max-chunk-size', type=int, default=4000,
                       help='Maximum chunk size (default: 4000)')
    parser.add_argument('--min-chunk-size', type=int, default=100,
                       help='Minimum chunk size (default: 100)')
    
    # Behavior options
    parser.add_argument('--preserve-sentences', action='store_true', default=True,
                       help='Try to preserve sentence boundaries (default: True)')
    parser.add_argument('--preserve-paragraphs', action='store_true',
                       help='Try to preserve paragraph boundaries')
    
    # Output options
    parser.add_argument('--output', type=Path,
                       help='Output file for chunks (JSON format)')
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Print chunk analysis (default: True)')
    parser.add_argument('--show-chunks', action='store_true',
                       help='Show individual chunk previews')
    
    # Tokenizer options
    parser.add_argument('--tokenizer-model', type=str, default='gpt-4',
                       help='Model for token counting (default: gpt-4)')
    
    return parser


def main() -> int:
    """Main function."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        if args.interactive:
            interactive_mode()
            return 0
        
        # Load input document
        if not args.input.exists():
            print(f"❌ Input file not found: {args.input}")
            return 1
        
        print(f"📂 Loading document: {args.input}")
        text = load_document(args.input)
        if not text:
            print("❌ Could not load document content")
            return 1
        
        print(f"📄 Document loaded: {len(text):,} characters")
        
        # Configure chunker
        config = ChunkingConfig(
            strategy=ChunkingStrategy(args.strategy),
            chunk_size=args.chunk_size,
            overlap_size=args.overlap,
            max_chunk_size=args.max_chunk_size,
            min_chunk_size=args.min_chunk_size,
            preserve_sentences=args.preserve_sentences,
            preserve_paragraphs=args.preserve_paragraphs,
            tokenizer_model=args.tokenizer_model
        )
        
        chunker = DocumentChunker(config)
        
        print(f"🔧 Chunking with {args.strategy} strategy...")
        chunks = chunker.chunk_document(text, args.input.stem)
        
        if args.analyze:
            print_chunk_analysis(chunks)
        
        if args.show_chunks:
            print(f"\n📄 Chunk previews:")
            for chunk in chunks[:5]:  # Show first 5
                preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                print(f"\n  Chunk {chunk.chunk_index} ({chunk.char_count} chars, {chunk.token_count} tokens):")
                print(f"    {preview}")
            
            if len(chunks) > 5:
                print(f"    ... and {len(chunks) - 5} more chunks")
        
        # Save output if requested
        if args.output:
            save_chunks(chunks, args.output)
        
        print(f"\n✅ Successfully chunked document into {len(chunks)} chunks")
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
