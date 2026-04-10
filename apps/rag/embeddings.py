#!/usr/bin/env python3
"""Generate embeddings from OpenAI or Azure OpenAI models.

Examples:
	python embeddings.py --text "Explain vector search"
	python embeddings.py --provider azure --text "Azure AI Search"
	python embeddings.py --input-file notes.txt --output embeddings.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
	from openai import OpenAI
except ImportError:
	OpenAI = None


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_AZURE_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class EmbeddingConfig:
	"""Configuration for the embedding generator."""

	provider: str = "openai"
	model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
	api_key: Optional[str] = None
	base_url: Optional[str] = None
	azure_endpoint: Optional[str] = None
	encoding_format: str = "float"
	dimensions: Optional[int] = None
	timeout: float = 60.0
	clean_text: bool = True
	min_text_length: int = 10
	collapse_whitespace: bool = True
	strip_urls: bool = False
	strip_emails: bool = False


@dataclass
class CleanedText:
	"""A cleaned embedding input and its provenance metadata."""

	original: str
	cleaned: str
	removed: bool = False
	reason: Optional[str] = None


_REPEATED_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HEADER_FOOTER_PATTERN = re.compile(r"^(page\s+\d+|copyright|all rights reserved|confidential)$", re.IGNORECASE)


def clean_embedding_text(text: str, config: EmbeddingConfig) -> CleanedText:
	"""Normalize noisy text before sending it for embedding."""

	original = text
	cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()

	if config.strip_urls:
		cleaned = _URL_PATTERN.sub(" ", cleaned)

	if config.strip_emails:
		cleaned = _EMAIL_PATTERN.sub(" ", cleaned)

	lines: list[str] = []
	for raw_line in cleaned.split("\n"):
		line = raw_line.strip()
		if not line:
			lines.append("")
			continue

		if _HEADER_FOOTER_PATTERN.fullmatch(line):
			continue

		if len(line) < 2 and not line.isalnum():
			continue

		lines.append(line)

	cleaned = "\n".join(lines)
	cleaned = _REPEATED_BLANK_LINES.sub("\n\n", cleaned)
	if config.collapse_whitespace:
		cleaned = _REPEATED_SPACES.sub(" ", cleaned)

	cleaned = cleaned.strip()
	if len(cleaned) < config.min_text_length:
		return CleanedText(
			original=original,
			cleaned=cleaned,
			removed=True,
			reason=f"cleaned text shorter than min_text_length={config.min_text_length}",
		)

	return CleanedText(original=original, cleaned=cleaned)


class EmbeddingGenerator:
	"""Small wrapper around the OpenAI embeddings API."""

	def __init__(self, config: EmbeddingConfig):
		self.config = config
		self.client = self._create_client()

	def _create_client(self) -> Any:
		if OpenAI is None:
			raise RuntimeError(
				"OpenAI library not found. Install dependencies with: pip install openai"
			)

		provider = self.config.provider.lower()
		if provider not in {"openai", "azure"}:
			raise ValueError(f"Unsupported provider: {provider}")

		if provider == "azure":
			base_url = self.config.base_url or os.getenv("AZURE_OPENAI_BASE_URL")
			azure_endpoint = self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
			if not base_url and azure_endpoint:
				base_url = azure_endpoint.rstrip("/") + "/openai/v1/"

			api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
			if not api_key:
				raise ValueError(
					"Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY or pass --api-key."
				)

			if not base_url:
				raise ValueError(
					"Missing Azure OpenAI endpoint. Set AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_BASE_URL."
				)

			return OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout)

		api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
		if not api_key:
			raise ValueError(
				"Missing OpenAI credentials. Set OPENAI_API_KEY or pass --api-key."
			)

		return OpenAI(
			api_key=api_key,
			base_url=self.config.base_url,
			timeout=self.config.timeout,
		)

	def generate(self, texts: list[str]) -> dict[str, Any]:
		if not texts:
			raise ValueError("At least one input text is required.")

		if self.config.clean_text:
			processed_inputs = [clean_embedding_text(text, self.config) for text in texts]
		else:
			processed_inputs = [CleanedText(original=text, cleaned=text.strip()) for text in texts]
		cleaned_inputs = [item for item in processed_inputs if not item.removed]
		if not cleaned_inputs:
			raise ValueError("All inputs were removed during cleaning. Lower --min-text-length or disable cleaning.")

		request: dict[str, Any] = {
			"model": self.config.model,
			"input": [item.cleaned for item in cleaned_inputs],
			"encoding_format": self.config.encoding_format,
		}
		if self.config.dimensions is not None:
			request["dimensions"] = self.config.dimensions

		response = self.client.embeddings.create(**request)
		return {
			"provider": self.config.provider,
			"model": response.model,
			"dimensions": len(response.data[0].embedding) if response.data else 0,
			"cleaning": {
				"enabled": self.config.clean_text,
				"min_text_length": self.config.min_text_length,
				"collapse_whitespace": self.config.collapse_whitespace,
				"strip_urls": self.config.strip_urls,
				"strip_emails": self.config.strip_emails,
				"removed_inputs": [
					{
						"original": item.original,
						"reason": item.reason,
					}
					for item in processed_inputs
					if item.removed
				],
			},
			"usage": {
				"prompt_tokens": response.usage.prompt_tokens,
				"total_tokens": response.usage.total_tokens,
			},
			"data": [
				{
					"index": item.index,
					"original_text": cleaned_inputs[item.index].original,
					"cleaned_text": cleaned_inputs[item.index].cleaned,
					"embedding": item.embedding,
				}
				for item in response.data
			],
		}


def _default_provider() -> str:
	provider = os.getenv("AI_PROVIDER", "openai").lower()
	return provider if provider in {"openai", "azure"} else "openai"


def _default_model(provider: str) -> str:
	if provider == "azure":
		return (
			os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
			or os.getenv("AZURE_OPENAI_MODEL")
			or os.getenv("EMBEDDING_MODEL")
			or DEFAULT_AZURE_EMBEDDING_MODEL
		)

	return os.getenv("EMBEDDING_MODEL") or DEFAULT_OPENAI_EMBEDDING_MODEL


def _build_config(args: argparse.Namespace) -> EmbeddingConfig:
	provider = args.provider or _default_provider()
	return EmbeddingConfig(
		provider=provider,
		model=args.model or _default_model(provider),
		api_key=args.api_key,
		base_url=args.base_url,
		azure_endpoint=args.azure_endpoint,
		encoding_format=args.encoding_format,
		dimensions=args.dimensions,
		timeout=args.timeout,
		clean_text=not args.no_clean,
		min_text_length=args.min_text_length,
		collapse_whitespace=not args.keep_whitespace,
		strip_urls=args.strip_urls,
		strip_emails=args.strip_emails,
	)


def _load_inputs(args: argparse.Namespace) -> list[str]:
	if args.text:
		return [args.text]

	if not args.input_file:
		raise ValueError("Provide --text or --input-file.")

	input_path = Path(args.input_file)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	raw_content = input_path.read_text(encoding="utf-8").strip()
	if not raw_content:
		raise ValueError("Input file is empty.")

	if args.batch_mode == "lines":
		lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
		if not lines:
			raise ValueError("Input file does not contain any non-empty lines.")
		return lines

	return [raw_content]


def _write_output(payload: dict[str, Any], output_path: Optional[str]) -> None:
	serialized = json.dumps(payload, indent=2)
	if output_path:
		Path(output_path).write_text(serialized + "\n", encoding="utf-8")
		print(f"Saved embeddings to {output_path}")
		return

	print(serialized)


def _print_summary(payload: dict[str, Any], preview_length: int) -> None:
	print(f"Provider: {payload.get('provider', '')}")
	print(f"Model: {payload.get('model', '')}")
	print(f"Dimensions: {payload.get('dimensions', 0)}")

	usage = payload.get("usage", {})
	print(f"Prompt tokens: {usage.get('prompt_tokens', 0)}")
	print(f"Total tokens: {usage.get('total_tokens', 0)}")

	removed_inputs = payload.get("cleaning", {}).get("removed_inputs", [])
	print(f"Removed inputs: {len(removed_inputs)}")

	data = payload.get("data", [])
	print(f"Returned embeddings: {len(data)}")

	for item in data:
		embedding = item.get("embedding", [])
		preview = embedding[:preview_length]
		print(f"\nItem {item.get('index', 0)}")
		print(f"Cleaned text: {item.get('cleaned_text', '')}")
		print(f"Vector length: {len(embedding)}")
		print(f"Vector preview ({len(preview)}): {preview}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Generate embeddings from OpenAI or Azure OpenAI.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Environment variables:
  AI_PROVIDER                      Provider (openai, azure)
  OPENAI_API_KEY                   OpenAI API key
  AZURE_OPENAI_API_KEY             Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT            Azure OpenAI resource endpoint
  AZURE_OPENAI_BASE_URL            Optional explicit Azure v1 base URL
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT Azure deployment name for embeddings
  EMBEDDING_MODEL                  Default embedding model
		""".strip(),
	)
	parser.add_argument("--text", help="Single text to embed.")
	parser.add_argument(
		"--input-file",
		help="File containing text to embed. Use --batch-mode lines to embed one line per item.",
	)
	parser.add_argument(
		"--batch-mode",
		choices=["document", "lines"],
		default="document",
		help="How to read --input-file. document embeds the whole file, lines embeds one non-empty line per item.",
	)
	parser.add_argument("--output", help="Optional output JSON file path.")
	parser.add_argument(
		"--summary",
		action="store_true",
		help="Print a short summary with vector length and a preview instead of full JSON.",
	)
	parser.add_argument(
		"--preview-length",
		type=int,
		default=5,
		help="How many embedding values to show in --summary mode.",
	)
	parser.add_argument("--provider", choices=["openai", "azure"], help="Embedding provider.")
	parser.add_argument("--model", help="Model or Azure deployment name.")
	parser.add_argument("--api-key", help="Override API key from environment variables.")
	parser.add_argument("--base-url", help="Optional custom OpenAI-compatible base URL.")
	parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint URL.")
	parser.add_argument(
		"--encoding-format",
		choices=["float", "base64"],
		default="float",
		help="Embedding encoding format returned by the API.",
	)
	parser.add_argument(
		"--dimensions",
		type=int,
		help="Optional dimensions override for models that support truncation.",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=60.0,
		help="Request timeout in seconds.",
	)
	parser.add_argument(
		"--no-clean",
		action="store_true",
		help="Disable text cleaning before embedding.",
	)
	parser.add_argument(
		"--min-text-length",
		type=int,
		default=10,
		help="Drop cleaned inputs shorter than this length.",
	)
	parser.add_argument(
		"--keep-whitespace",
		action="store_true",
		help="Preserve repeated spaces instead of collapsing them.",
	)
	parser.add_argument(
		"--strip-urls",
		action="store_true",
		help="Remove URLs before embedding.",
	)
	parser.add_argument(
		"--strip-emails",
		action="store_true",
		help="Remove email addresses before embedding.",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	try:
		texts = _load_inputs(args)
		generator = EmbeddingGenerator(_build_config(args))
		payload = generator.generate(texts)
		if args.summary:
			_print_summary(payload, max(args.preview_length, 0))
		else:
			_write_output(payload, args.output)
		return 0
	except KeyboardInterrupt:
		print("Cancelled by user.")
		return 130
	except Exception as exc:
		print(f"Error: {exc}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
