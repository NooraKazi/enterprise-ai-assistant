"""Prompt catalog and helpers for controlled model responses."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, Optional


@dataclass(frozen=True)
class PromptPreset:
	"""Named prompt instruction for tone or output format."""

	name: str
	instruction: str
	description: str


@dataclass(frozen=True)
class PromptTemplate:
	"""Reusable task-specific system prompt template."""

	name: str
	file_path: str
	instruction: str
	description: str
	default_response_format: str = "text"


BASE_SYSTEM_PROMPT = (
	"You are the core assistant for an enterprise AI project. "
	"Give accurate, practical, and well-structured answers."
)


PROMPTS_DIR = Path(__file__).with_name("prompts")


# Metadata keeps routing labels and response defaults separate from the actual
# prompt text files so prompt content can stay focused on instructions.
PROMPT_METADATA: Dict[str, Dict[str, object]] = {
	"chatbot_personality": {
		"description": "General assistant personality for everyday chat and support.",
		"default_response_format": "text",
	},
	"summarizer": {
		"description": "Condenses long content into a clear summary.",
		"default_response_format": "text",
	},
	"rag": {
		"description": "Grounded question answering for retrieval-augmented generation.",
		"default_response_format": "text",
	},
	"json_extractor": {
		"description": "Extracts structured fields and returns strict JSON.",
		"default_response_format": "json_object",
	},
	"code_generator": {
		"description": "Generates implementation-focused code and technical solutions.",
		"default_response_format": "text",
	},
}


def _normalize_template_name(value: str) -> str:
	"""Normalize prompt file names and aliases into stable template keys."""
	normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
	if normalized.endswith("_prompt"):
		normalized = normalized[:-7]
	return normalized


@lru_cache(maxsize=1)
def _discover_prompt_files() -> Dict[str, Path]:
	"""Map normalized template keys to files in the prompts directory."""
	if not PROMPTS_DIR.exists():
		return {}

	file_map: Dict[str, Path] = {}
	for path in sorted(PROMPTS_DIR.iterdir()):
		if not path.is_file():
			continue

		# Support mixed file naming styles such as rag_prompt.txt and json_extractor.
		lookup_name = path.stem if path.suffix else path.name
		file_map[_normalize_template_name(lookup_name)] = path

	return file_map


@lru_cache(maxsize=1)
def _load_prompt_library() -> Dict[str, PromptTemplate]:
	"""Load reusable prompt templates from disk."""
	library: Dict[str, PromptTemplate] = {}
	for template_name, metadata in PROMPT_METADATA.items():
		prompt_file = _discover_prompt_files().get(template_name)
		if not prompt_file:
			continue

		library[template_name] = PromptTemplate(
			name=template_name,
			file_path=str(prompt_file),
			instruction=prompt_file.read_text(encoding="utf-8").strip(),
			description=str(metadata["description"]),
			default_response_format=str(metadata.get("default_response_format", "text")),
		)

	return library


# Tone presets shape style, while format presets shape response structure.
TONE_PRESETS: Dict[str, PromptPreset] = {
	"balanced": PromptPreset(
		name="balanced",
		instruction="Use a clear, professional, and helpful tone.",
		description="Default professional tone.",
	),
	"concise": PromptPreset(
		name="concise",
		instruction="Be brief, direct, and avoid unnecessary detail.",
		description="Short and direct answers.",
	),
	"friendly": PromptPreset(
		name="friendly",
		instruction="Use a warm, approachable, and encouraging tone.",
		description="Approachable and easy to read.",
	),
	"technical": PromptPreset(
		name="technical",
		instruction="Use precise technical language and explain implementation details carefully.",
		description="Engineering-focused explanations.",
	),
	"executive": PromptPreset(
		name="executive",
		instruction="Sound concise, decisive, and outcome-focused for a business audience.",
		description="High-level business-ready responses.",
	),
	"teacher": PromptPreset(
		name="teacher",
		instruction="Explain concepts step by step and define important terms simply.",
		description="Learning-oriented explanations.",
	),
}


FORMAT_PRESETS: Dict[str, PromptPreset] = {
	"paragraph": PromptPreset(
		name="paragraph",
		instruction="Respond in short, well-structured paragraphs.",
		description="Paragraph-based response.",
	),
	"bullet": PromptPreset(
		name="bullet",
		instruction="Respond as a concise bullet list.",
		description="Bullet list output.",
	),
	"steps": PromptPreset(
		name="steps",
		instruction="Respond as numbered steps in execution order.",
		description="Step-by-step instructions.",
	),
	"summary": PromptPreset(
		name="summary",
		instruction="Start with a short summary, then give the key supporting points.",
		description="Summary-first response.",
	),
	"table": PromptPreset(
		name="table",
		instruction="When the content fits, respond using a compact Markdown table; otherwise explain why a table is not suitable and use bullets.",
		description="Prefer tabular output.",
	),
}


def list_tones() -> list[str]:
	"""Return supported tone preset names."""
	return sorted(TONE_PRESETS.keys())


def list_formats() -> list[str]:
	"""Return supported format preset names."""
	return sorted(FORMAT_PRESETS.keys())


def list_prompt_templates() -> list[str]:
	"""Return supported prompt template names."""
	return sorted(_load_prompt_library().keys())


def list_prompt_template_choices() -> list[str]:
	"""Return supported CLI choices including automatic routing."""
	return ["auto", *list_prompt_templates()]


def get_prompt_template(template: Optional[str]) -> Optional[PromptTemplate]:
	"""Resolve a prompt template by name."""
	if not template:
		return None

	if template == "auto":
		return None

	return _load_prompt_library().get(_normalize_template_name(template))


def infer_prompt_template(question: str) -> PromptTemplate:
	"""Infer the best prompt template for a user request using simple intent rules."""
	text = question.strip().lower()
	library = _load_prompt_library()
	if not text:
		return library["chatbot_personality"]

	# Keep routing intentionally lightweight so prompt selection stays transparent.
	scores: Dict[str, int] = {name: 0 for name in library}

	for keyword in ("summarize", "summary", "summarise", "recap", "tl;dr", "bullet points"):
		if keyword in text:
			scores["summarizer"] += 3

	for keyword in ("context:", "based on the provided context", "use the provided context", "using the context", "cite", "citation", "source"):
		if keyword in text:
			scores["rag"] += 3

	for keyword in ("extract", "json", "structured data", "fields", "invoice", "receipt", "parse"):
		if keyword in text:
			scores["json_extractor"] += 3

	for keyword in ("write code", "generate code", "function", "class", "script", "api", "endpoint", "refactor", "implement", "python", "javascript", "typescript", "java", "sql"):
		if keyword in text:
			scores["code_generator"] += 3

	if any(token in text for token in ("return valid json", "json only", "extract structured")):
		scores["json_extractor"] += 4

	if any(token in text for token in ("write a", "implement a", "create a function", "generate a function")):
		scores["code_generator"] += 2

	best_template = max(scores.items(), key=lambda item: item[1])[0]
	if scores[best_template] == 0:
		best_template = "chatbot_personality"

	return library[best_template]


def resolve_prompt_template(question: str, template: Optional[str] = None) -> PromptTemplate:
	"""Resolve an explicit prompt template or infer one from the request intent."""
	explicit_template = get_prompt_template(template)
	if explicit_template:
		return explicit_template

	return infer_prompt_template(question)


def build_system_prompt(
	custom_system_prompt: Optional[str] = None,
	tone: str = "balanced",
	output_format: str = "paragraph",
	template: Optional[str] = None,
) -> str:
	"""Build a single system prompt from reusable prompt presets."""
	# The final system prompt is composed in layers so the base assistant rules,
	# task template, style, and optional user override remain explicit.
	tone_preset = TONE_PRESETS.get(tone, TONE_PRESETS["balanced"])
	format_preset = FORMAT_PRESETS.get(output_format, FORMAT_PRESETS["paragraph"])
	# The resolved file-backed template becomes the base task instruction for the turn.
	template_preset = resolve_prompt_template("", template) if template else _load_prompt_library().get("chatbot_personality")

	parts = [
		BASE_SYSTEM_PROMPT,
	]

	if template_preset:
		parts.append(template_preset.instruction)

	parts.extend([
		tone_preset.instruction,
		format_preset.instruction,
	])

	if custom_system_prompt:
		parts.append(custom_system_prompt.strip())

	return "\n\n".join(parts)


def describe_presets() -> str:
	"""Return a readable overview of available prompt presets."""
	template_lines = [
		f"  {name:<20} - {template.description} ({Path(template.file_path).name})"
		for name, template in sorted(_load_prompt_library().items())
	]
	tone_lines = [f"  {name:<10} - {preset.description}" for name, preset in sorted(TONE_PRESETS.items())]
	format_lines = [f"  {name:<10} - {preset.description}" for name, preset in sorted(FORMAT_PRESETS.items())]

	return "\n".join([
		"Available prompt templates:",
		*template_lines,
		"",
		"Available tones:",
		*tone_lines,
		"",
		"Available formats:",
		*format_lines,
	])
