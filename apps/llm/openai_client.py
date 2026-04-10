#!/usr/bin/env python3
"""
Enterprise AI Assistant - Core Brain LLM Client
A simple CLI tool to ask anything and get AI responses
Supports OpenAI, Azure OpenAI (Microsoft Foundry), and GitHub Models
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import asdict, dataclass, field
import requests
from enum import Enum

from prompts import (
    build_system_prompt,
    describe_presets,
    list_formats,
    list_prompt_template_choices,
    list_prompt_templates,
    list_tones,
    resolve_prompt_template,
)

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI library not found. Install with: pip install openai")
    sys.exit(1)


CHAT_STATE_FILE = Path(__file__).with_name(".chat_state.json")
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_AZURE_EMBEDDING_MODEL = "text-embedding-3-small"


class ResponseFormat(Enum):
    """Supported response formats"""
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass
class ChatTurn:
    """A single conversational turn captured by the CLI chatbot."""

    user_message: str
    assistant_message: str
    prompt_template: str
    prompt_file: str


@dataclass
class ClientConfig:
    """Configuration for different AI providers"""
    # Provider-specific connection settings live here so CLI flags and env vars
    # can be merged into one runtime config object.
    provider: str = "openai"  # openai, azure, github
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4.1-nano"
    azure_endpoint: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    max_tokens: int = 1000
    temperature: float = 0.7
    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: Optional[Dict[str, Any]] = None
    stream: bool = False
    prompt_template: Optional[str] = "auto"
    last_prompt_template: Optional[str] = None
    last_prompt_file: Optional[str] = None
    system_prompt: Optional[str] = None
    # These fields keep the interactive CLI in chatbot mode instead of one-shot mode.
    chat_history: list[ChatTurn] = field(default_factory=list)
    conversation_items: list[Dict[str, Any]] = field(default_factory=list)
    use_chat_history: bool = True
    persist_chat_history: bool = True
    chat_state_file: str = str(CHAT_STATE_FILE)
    tone: str = "balanced"
    output_format: str = "paragraph"
    show_embeddings: bool = False
    embedding_preview_length: int = 5
    embedding_model: Optional[str] = None


class LLMClient:
    """Universal LLM client supporting multiple providers"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = None
        self._setup_client()
        self._load_chat_state()
    
    def _setup_client(self):
        """Initialize the appropriate client based on provider"""
        # Each provider uses the same SDK client, but the base URL and auth
        # source differ depending on the backend.
        if self.config.provider == "openai":
            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY")
            )
        elif self.config.provider == "azure":
            azure_endpoint = self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_base_url = self.config.base_url or os.getenv("AZURE_OPENAI_BASE_URL")

            if not azure_base_url and azure_endpoint:
                azure_base_url = azure_endpoint.rstrip("/") + "/openai/v1/"

            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                base_url=azure_base_url,
            )
        elif self.config.provider == "github":
            self.client = OpenAI(
                api_key=self.config.api_key or os.getenv("GITHUB_TOKEN"),
                base_url="https://models.github.ai/inference/"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _build_input(self, question: str, system_prompt: Optional[str] = None) -> Union[str, list[Dict[str, Any]]]:
        """Build Responses API input payload with optional conversation history."""
        user_item = self._build_message_item("user", question)
        if not self.config.use_chat_history:
            if not system_prompt:
                return [user_item]
            return [self._build_message_item("system", system_prompt), user_item]

        # Conversation items are replayed each turn so the CLI behaves like a chatbot.
        conversation = list(self.config.conversation_items)
        if system_prompt:
            system_item = self._build_message_item("system", system_prompt)
            if not conversation:
                conversation.append(system_item)
            elif conversation[0].get("role") != "system":
                # Stored turns begin with the first user message, so prepend the current system prompt.
                conversation.insert(0, system_item)
            elif conversation[0] != system_item:
                # Only reset when a previously stored system prompt is explicitly different.
                conversation = [system_item, *conversation[1:]]

        return [*conversation, user_item]

    def _load_chat_state(self):
        """Load persisted chatbot history from disk when enabled."""
        if not self.config.persist_chat_history:
            return

        state_path = Path(self.config.chat_state_file)
        if not state_path.exists():
            return

        try:
            stored_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if self.config.system_prompt is None:
            self.config.system_prompt = stored_state.get("system_prompt")

        self.config.chat_history = [
            ChatTurn(**turn)
            for turn in stored_state.get("chat_history", [])
            if isinstance(turn, dict)
        ]

        conversation_items = stored_state.get("conversation_items", [])
        if isinstance(conversation_items, list):
            self.config.conversation_items = [
                self._normalize_message_item(item)
                for item in conversation_items
                if isinstance(item, dict)
            ]

    def _save_chat_state(self):
        """Persist chatbot history to disk so it survives process restarts."""
        if not self.config.persist_chat_history:
            return

        state_path = Path(self.config.chat_state_file)
        payload = {
            "system_prompt": self.config.system_prompt,
            "chat_history": [asdict(turn) for turn in self.config.chat_history],
            "conversation_items": self.config.conversation_items,
        }

        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _resolve_embedding_model(self) -> Optional[str]:
        if self.config.embedding_model:
            return self.config.embedding_model

        if self.config.provider == "azure":
            return (
                os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                or os.getenv("AZURE_OPENAI_MODEL")
                or os.getenv("EMBEDDING_MODEL")
                or DEFAULT_AZURE_EMBEDDING_MODEL
            )

        if self.config.provider == "openai":
            return os.getenv("EMBEDDING_MODEL") or DEFAULT_OPENAI_EMBEDDING_MODEL

        return None

    def _build_embedding_client(self):
        if self.config.provider == "openai":
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI credentials for embedding preview.")
            return OpenAI(api_key=api_key)

        if self.config.provider == "azure":
            azure_endpoint = self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_base_url = self.config.base_url or os.getenv("AZURE_OPENAI_BASE_URL")
            if not azure_base_url and azure_endpoint:
                azure_base_url = azure_endpoint.rstrip("/") + "/openai/v1/"

            api_key = self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing Azure OpenAI credentials for embedding preview.")
            if not azure_base_url:
                raise ValueError("Missing Azure OpenAI endpoint for embedding preview.")

            return OpenAI(api_key=api_key, base_url=azure_base_url)

        raise ValueError(f"Embedding preview is not supported for provider: {self.config.provider}")

    def _print_embedding_summary(self, assistant_message: str) -> None:
        if not self.config.show_embeddings:
            return

        if not assistant_message or assistant_message.startswith("❌"):
            return

        embedding_model = self._resolve_embedding_model()
        if not embedding_model:
            print("📐 Embeddings: skipped, no embedding model configured")
            return

        try:
            embedding_client = self._build_embedding_client()
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=[assistant_message],
                encoding_format="float",
            )
            embedding = response.data[0].embedding if response.data else []
            preview_length = max(self.config.embedding_preview_length, 0)
            preview = embedding[:preview_length]

            print("📐 Embedding summary:")
            print(f"  Model: {response.model}")
            print(f"  Dimensions: {len(embedding)}")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Preview ({len(preview)}): {preview}")
        except Exception as exc:
            print(f"📐 Embeddings: failed to generate summary ({exc})")

    def _build_message_item(self, role: str, text: str) -> Dict[str, Any]:
        """Create a Responses API message item."""
        content_type = "output_text" if role == "assistant" else "input_text"
        return {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        }

    def _normalize_message_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize loaded history items to the content types expected by the Responses API."""
        role = item.get("role")
        content_type = "output_text" if role == "assistant" else "input_text"
        normalized_item = dict(item)
        normalized_content = []

        for content_item in item.get("content", []):
            if not isinstance(content_item, dict):
                continue

            normalized_content.append(
                {
                    "type": content_type,
                    "text": content_item.get("text", ""),
                }
            )

        if not normalized_content:
            normalized_content = [{"type": content_type, "text": ""}]

        normalized_item["content"] = normalized_content
        return normalized_item

    def clear_chat_history(self):
        """Reset the active chatbot conversation."""
        self.config.chat_history.clear()
        self.config.conversation_items.clear()
        self._save_chat_state()

    def has_chat_history(self) -> bool:
        """Return whether a previous conversation exists."""
        return bool(self.config.chat_history or self.config.conversation_items)

    def _record_chat_turn(
        self,
        question: str,
        assistant_message: str,
        prompt_template: str,
        prompt_file: str,
        response: Optional[Any] = None,
    ):
        """Persist conversation history for subsequent chatbot turns."""
        if not self.config.use_chat_history:
            return

        self.config.conversation_items.append(self._build_message_item("user", question))
        self.config.conversation_items.append(self._build_message_item("assistant", assistant_message))

        self.config.chat_history.append(
            ChatTurn(
                user_message=question,
                assistant_message=assistant_message,
                prompt_template=prompt_template,
                prompt_file=prompt_file,
            )
        )
        self._save_chat_state()

    def _build_text_config(
        self,
        response_format: ResponseFormat,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build Responses API text formatting configuration."""
        if response_format == ResponseFormat.JSON_OBJECT:
            return {"format": {"type": "json_object"}}

        if response_format == ResponseFormat.JSON_SCHEMA and json_schema:
            return {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                }
            }

        return None

    def _resolve_response_format(
        self,
        requested_format: Optional[ResponseFormat],
        prompt_template_name: Optional[str],
    ) -> ResponseFormat:
        """Resolve the effective response format for a request."""
        if requested_format:
            return requested_format

        if prompt_template_name == "json_extractor":
            return ResponseFormat.JSON_OBJECT

        return self.config.response_format
    
    def ask(self, 
           question: str, 
           system_prompt: Optional[str] = None,
           response_format: Optional[ResponseFormat] = None,
           json_schema: Optional[Dict[str, Any]] = None,
           prompt_template: Optional[str] = None,
           tone: Optional[str] = None,
           output_format: Optional[str] = None,
           stream: bool = False) -> Union[str, Dict[str, Any]]:
        """Ask a question using the OpenAI Responses API."""
        try:
            template_selection = prompt_template if prompt_template is not None else self.config.prompt_template
            # Resolve the best reusable prompt file before constructing the turn instructions.
            resolved_template = resolve_prompt_template(question, template_selection)
            self.config.last_prompt_template = resolved_template.name
            self.config.last_prompt_file = Path(resolved_template.file_path).name

            if self.client is None:
                raise RuntimeError("LLM client is not initialized")

            format_to_use = self._resolve_response_format(response_format, resolved_template.name)
            schema_to_use = json_schema or self.config.json_schema
            custom_system_prompt = system_prompt if system_prompt is not None else self.config.system_prompt
            resolved_system_prompt = build_system_prompt(
                custom_system_prompt=custom_system_prompt,
                tone=tone or self.config.tone,
                output_format=output_format or self.config.output_format,
                template=resolved_template.name,
            )

            api_params = {
                "model": self.config.model,
                "input": self._build_input(question, resolved_system_prompt),
                "max_output_tokens": self.config.max_tokens,
            }

            text_config = self._build_text_config(format_to_use, schema_to_use)
            if text_config:
                api_params["text"] = text_config

            if stream or self.config.stream:
                streamed_text, final_response = self._handle_streaming_response(api_params)
                final_result = self._coerce_response_output(streamed_text, format_to_use)
                self._record_chat_turn(
                    question=question,
                    assistant_message=self._stringify_response(final_result),
                    prompt_template=resolved_template.name,
                    prompt_file=self.config.last_prompt_file or "",
                    response=final_response,
                )
                self._print_embedding_summary(self._stringify_response(final_result))
                return final_result

            response = self.client.responses.create(**api_params)
            result = self._process_response(response, format_to_use)
            self._record_chat_turn(
                question=question,
                assistant_message=self._stringify_response(result),
                prompt_template=resolved_template.name,
                prompt_file=self.config.last_prompt_file or "",
                response=response,
            )
            self._print_embedding_summary(self._stringify_response(result))
            return result
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _stringify_response(self, response: Union[str, Dict[str, Any]]) -> str:
        """Convert a response payload into a history-safe string."""
        if isinstance(response, dict):
            return json.dumps(response, indent=2)
        return response

    def _coerce_response_output(
        self,
        content: str,
        response_format: ResponseFormat,
    ) -> Union[str, Dict[str, Any]]:
        """Convert raw output text into the configured response shape."""
        if response_format in [ResponseFormat.JSON_OBJECT, ResponseFormat.JSON_SCHEMA]:
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                return f"❌ Invalid JSON response: {content}"

        return content.strip()
    
    def _process_response(self, response, response_format: ResponseFormat) -> Union[str, Dict[str, Any]]:
        """Process the Responses API payload."""
        content = getattr(response, "output_text", None)
        
        if not content:
            return "❌ Empty response received"
            
        # Parse JSON responses
        if response_format in [ResponseFormat.JSON_OBJECT, ResponseFormat.JSON_SCHEMA]:
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                return f"❌ Invalid JSON response: {content}"
        
        return content.strip()
    
    def _handle_streaming_response(self, api_params: Dict[str, Any]) -> tuple[str, Any]:
        """Handle streaming responses from the Responses API."""
        try:
            full_response = ""
            if self.client is None:
                raise RuntimeError("LLM client is not initialized")

            client = self.client
            
            print("\n🤖 AI: ", end="", flush=True)
            with client.responses.stream(**api_params) as response_stream:
                for event in response_stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                        full_response += event.delta

                response_stream.until_done()
                final_response = response_stream.get_final_response()
            print()  # New line after streaming
            
            return full_response, final_response
            
        except Exception as e:
            return f"❌ Streaming error: {str(e)}", None
    
    def get_available_models(self) -> list:
        """Get list of available models (GitHub Models only for now)"""
        if self.config.provider == "github":
            try:
                response = requests.get("https://models.github.ai/catalog/models")
                if response.status_code == 200:
                    models = response.json()
                    return [model.get("id", "Unknown") for model in models]
            except:
                pass
        return [self.config.model]


def load_config() -> ClientConfig:
    """Load configuration from environment variables and defaults"""
    config = ClientConfig()
    
    # Resolve provider first so the model and endpoint defaults can follow it.
    provider = os.getenv("AI_PROVIDER", "openai").lower()
    config.provider = provider
    
    # Set model based on provider
    if provider == "openai":
        config.model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    elif provider == "azure":
        config.model = os.getenv("AZURE_OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-nano")
        config.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        config.base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    elif provider == "github":
        config.model = os.getenv("GITHUB_MODEL", "gpt-4o-mini")
    
    # Common settings
    config.max_tokens = int(os.getenv("AI_MAX_TOKENS", "1000"))
    config.temperature = float(os.getenv("AI_TEMPERATURE", "0.7"))
    config.stream = os.getenv("AI_STREAM", "false").lower() == "true"
    config.prompt_template = os.getenv("AI_PROMPT_TEMPLATE", "auto").lower()
    config.system_prompt = os.getenv("AI_SYSTEM_PROMPT")
    config.chat_state_file = os.getenv("AI_CHAT_STATE_FILE", str(CHAT_STATE_FILE))
    config.persist_chat_history = os.getenv("AI_PERSIST_CHAT_HISTORY", "true").lower() == "true"
    config.tone = os.getenv("AI_TONE", "balanced").lower()
    config.output_format = os.getenv("AI_OUTPUT_FORMAT", "paragraph").lower()
    config.show_embeddings = os.getenv("AI_SHOW_EMBEDDINGS", "false").lower() == "true"
    config.embedding_preview_length = int(os.getenv("AI_EMBEDDING_PREVIEW_LENGTH", "5"))
    if provider == "azure":
        config.embedding_model = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_MODEL")
            or os.getenv("EMBEDDING_MODEL")
        )
    elif provider == "openai":
        config.embedding_model = os.getenv("EMBEDDING_MODEL")
    
    # Response format settings
    format_str = os.getenv("AI_RESPONSE_FORMAT", "text").lower()
    if format_str == "json":
        config.response_format = ResponseFormat.JSON_OBJECT
    elif format_str == "json_schema":
        config.response_format = ResponseFormat.JSON_SCHEMA
    
    return config


def interactive_mode(client: LLMClient):
    """Interactive chat mode"""
    print("🧠 Enterprise AI Assistant - Interactive Mode")
    print("💡 Type 'quit', 'exit', or 'q' to stop")
    print("💡 Type 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            # Give the user a lightweight chance to reset the conversation before
            # every turn without forcing them to restart the program.
            if client.config.persist_chat_history:
                while True:
                    session_choice = input("\n🆕 Start a new chat session? [y/N]: ").strip().lower()

                    if session_choice in {"", "n", "no"}:
                        break

                    if session_choice in {"y", "yes"}:
                        client.clear_chat_history()
                        print("🧹 Started a new chat session")
                        break

                    if session_choice in {"quit", "exit", "q"}:
                        print("👋 Goodbye!")
                        return

                    print("❌ Please answer with 'y' or 'n'.")

            question = input("\n🤔 Ask anything: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n📋 Available commands:")
                print("  help       - Show this help")
                print("  models     - List available models")
                print("  config     - Show current configuration")
                print("  prompts    - Show available prompt, tone, and format presets")
                print("  template <t> - Change the active prompt template or set auto")
                print("  system <p> - Set a custom system prompt and reset chat history")
                print("  system clear - Remove the custom system prompt and reset history")
                print("  tone <t>   - Change the tone preset")
                print("  format <f> - Change the output format preset")
                print("  history    - Show the current chat history")
                print("  clear      - Clear chat history")
                print("  json <q>   - Ask question with JSON response")
                print("  stream <q> - Ask question with streaming response")
                print("  quit/q     - Exit the program")
                continue
            elif question.lower() == 'models':
                models = client.get_available_models()
                print(f"\n📋 Available models: {', '.join(models)}")
                continue
            elif question.lower() == 'config':
                print(f"\n⚙️ Current configuration:")
                print(f"  Provider: {client.config.provider}")
                print(f"  Model: {client.config.model}")
                print(f"  Max tokens: {client.config.max_tokens}")
                print(f"  Temperature: {client.config.temperature}")
                print(f"  Response format: {client.config.response_format.value}")
                print(f"  Streaming: {client.config.stream}")
                print(f"  Prompt template: {client.config.prompt_template}")
                print(f"  Last prompt template: {client.config.last_prompt_template or 'N/A'}")
                print(f"  Last prompt file: {client.config.last_prompt_file or 'N/A'}")
                print(f"  Custom system prompt: {client.config.system_prompt or 'N/A'}")
                print(f"  Chat turns: {len(client.config.chat_history)}")
                print(f"  Persist chat history: {client.config.persist_chat_history}")
                print(f"  Chat state file: {client.config.chat_state_file}")
                print(f"  Tone: {client.config.tone}")
                print(f"  Output format: {client.config.output_format}")
                continue
            elif question.lower() == 'prompts':
                print(f"\n{describe_presets()}")
                continue
            elif question.lower() == 'history':
                if not client.config.chat_history:
                    print("\n🗂️ No chat history yet")
                    continue

                print("\n🗂️ Chat history:")
                for index, turn in enumerate(client.config.chat_history, start=1):
                    print(f"  [{index}] You: {turn.user_message}")
                    print(f"      AI: {turn.assistant_message}")
                    print(f"      Prompt: {turn.prompt_file}")
                continue
            elif question.lower() == 'clear':
                client.clear_chat_history()
                print("\n🧹 Chat history cleared")
                continue
            elif question.lower() == 'system clear':
                client.config.system_prompt = None
                client.clear_chat_history()
                print("\n✅ Custom system prompt cleared and chat history reset")
                continue
            elif question.lower().startswith('system '):
                client.config.system_prompt = question[7:].strip()
                client.clear_chat_history()
                print("\n✅ Custom system prompt set and chat history reset")
                continue
            elif question.lower().startswith('template '):
                selected_template = question[9:].strip().lower()
                if selected_template not in list_prompt_template_choices():
                    print(f"\n❌ Unknown template '{selected_template}'. Available templates: {', '.join(list_prompt_template_choices())}")
                    continue
                client.config.prompt_template = selected_template
                client.clear_chat_history()
                print(f"\n✅ Prompt template set to {selected_template}")
                continue
            elif question.lower().startswith('tone '):
                selected_tone = question[5:].strip().lower()
                if selected_tone not in list_tones():
                    print(f"\n❌ Unknown tone '{selected_tone}'. Available tones: {', '.join(list_tones())}")
                    continue
                client.config.tone = selected_tone
                client.clear_chat_history()
                print(f"\n✅ Tone set to {selected_tone}")
                continue
            elif question.lower().startswith('format '):
                selected_format = question[7:].strip().lower()
                if selected_format not in list_formats():
                    print(f"\n❌ Unknown format '{selected_format}'. Available formats: {', '.join(list_formats())}")
                    continue
                client.config.output_format = selected_format
                client.clear_chat_history()
                print(f"\n✅ Output format set to {selected_format}")
                continue
            elif question.lower().startswith('json '):
                # Handle JSON mode requests
                actual_question = question[5:]  # Remove 'json ' prefix
                print("\n🤖 AI (JSON): ", end="", flush=True)
                response = client.ask(actual_question, response_format=ResponseFormat.JSON_OBJECT)
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(response)
                continue
            elif question.lower().startswith('stream '):
                # Handle streaming requests
                actual_question = question[7:]  # Remove 'stream ' prefix
                response = client.ask(actual_question, stream=True)
                continue
            
            if client.config.stream:
                # Streaming is enabled by default
                response = client.ask(question, stream=True)
            else:
                print("\n🤖 AI: ", end="", flush=True)
                response = client.ask(question)
                print(response)

            if client.config.last_prompt_file:
                print(f"🧩 Prompt file: {client.config.last_prompt_file}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Enterprise AI Assistant - Ask anything CLI tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python openai_client.py
  
  # Single question
  python openai_client.py -q "What is Python?"

    # Interactive chatbot with history
    python openai_client.py -p azure
  
  # JSON response format
  python openai_client.py --json -q "List 3 colors as JSON"

    # Use the JSON extractor prompt library
    python openai_client.py --template json_extractor -q "Extract the customer, invoice id, and due date from: Invoice INV-102 for Contoso is due on 2026-05-01"

    # Technical tone with bullet output
    python openai_client.py --tone technical --format bullet -q "Explain retrieval augmented generation"
  
  # Streaming response
  python openai_client.py --stream -q "Tell me a story"
  
  # Use different provider
  python openai_client.py -p azure -q "Explain AI"
  
  # Use specific model with streaming
  python openai_client.py -m gpt-4 --stream -q "Write a poem"

Environment Variables:
  AI_PROVIDER          Provider (openai, azure, github)
  OPENAI_API_KEY       OpenAI API key
  AZURE_OPENAI_API_KEY Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT Azure OpenAI endpoint
  GITHUB_TOKEN         GitHub personal access token
  AI_MAX_TOKENS        Maximum response tokens (default: 1000)
  AI_TEMPERATURE       Response creativity (0.0-1.0, default: 0.7)
  AI_RESPONSE_FORMAT   Default format (text, json, json_schema)
  AI_STREAM            Enable streaming (true/false)
    AI_SHOW_EMBEDDINGS   Print embedding summary after each chat turn (true/false)
    AI_EMBEDDING_PREVIEW_LENGTH Number of embedding values to preview (default: 5)

Interactive Commands:
  help                 Show available commands
  models              List available models
  config              Show current configuration
    prompts             Show prompt, tone, and format presets
    template <template> Set the active prompt template
    system <prompt>     Set a custom system prompt and reset chat history
    system clear        Clear the custom system prompt and reset chat history
    history             Show chat history
    clear               Clear chat history
    tone <tone>         Set the active tone preset
    format <format>     Set the active output format preset
  json <question>     Ask with JSON response format
  stream <question>   Ask with streaming response
  quit/q              Exit the program
        """
    )
    
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    parser.add_argument("-p", "--provider", choices=['openai', 'azure', 'github'], 
                       help="AI provider to use")
    parser.add_argument("-m", "--model", help="Model to use")
    parser.add_argument("-s", "--system", help="System prompt")
    parser.add_argument("--template", choices=list_prompt_template_choices(), help="Reusable prompt template or auto intent routing")
    parser.add_argument("--tone", choices=list_tones(), help="Tone preset for the response")
    parser.add_argument("--format", dest="output_format", choices=list_formats(), help="Output format preset")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens in response")
    parser.add_argument("--temperature", type=float, help="Response creativity (0.0-1.0)")
    parser.add_argument("--json", action="store_true", help="Request JSON response format")
    parser.add_argument("--stream", action="store_true", help="Enable streaming response")
    parser.add_argument("--show-embeddings", action="store_true", help="Print an embedding summary after each assistant response")
    parser.add_argument("--embedding-preview-length", type=int, help="How many embedding values to preview after each chat turn")
    parser.add_argument("--config", action="store_true", help="Show configuration and exit")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.temperature:
        config.temperature = args.temperature
    if args.template:
        config.prompt_template = args.template
    if args.system:
        config.system_prompt = args.system
    if args.tone:
        config.tone = args.tone
    if args.output_format:
        config.output_format = args.output_format
    if args.json:
        config.response_format = ResponseFormat.JSON_OBJECT
    if args.stream:
        config.stream = True
    if args.show_embeddings:
        config.show_embeddings = True
    if args.embedding_preview_length is not None:
        config.embedding_preview_length = args.embedding_preview_length
    
    # Show config and exit if requested
    if args.config:
        print("⚙️ Current Configuration:")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
        print(f"  Max tokens: {config.max_tokens}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Response format: {config.response_format.value}")
        print(f"  Streaming: {config.stream}")
        print(f"  Prompt template: {config.prompt_template}")
        print(f"  Last prompt template: {config.last_prompt_template or 'N/A'}")
        print(f"  Last prompt file: {config.last_prompt_file or 'N/A'}")
        print(f"  Custom system prompt: {config.system_prompt or 'N/A'}")
        print(f"  Chat turns: {len(config.chat_history)}")
        print(f"  Persist chat history: {config.persist_chat_history}")
        print(f"  Chat state file: {config.chat_state_file}")
        print(f"  Tone: {config.tone}")
        print(f"  Output format: {config.output_format}")
        print(f"  Show embeddings: {config.show_embeddings}")
        print(f"  Embedding preview length: {config.embedding_preview_length}")
        print(f"  Embedding model: {config.embedding_model or 'N/A'}")
        return
    
    # Initialize client
    try:
        client = LLMClient(config)
        print(f"✅ Connected to {config.provider.upper()} ({config.model})")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        print("\n💡 Make sure you have set the appropriate environment variables:")
        if config.provider == "openai":
            print("  $env:OPENAI_API_KEY=your_key_here")
        elif config.provider == "azure":
            print("  $env:AZURE_OPENAI_API_KEY=your_key_here")
            print("  $env:AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("  $env:AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
        elif config.provider == "github":
            print("  $env:GITHUB_TOKEN=your_token_here")
        sys.exit(1)
    
    # Handle single question or interactive mode
    if args.question:
        print(f"🤔 Question: {args.question}")
        
        # Determine response format
        response_format = None
        if config.response_format == ResponseFormat.JSON_OBJECT:
            response_format = ResponseFormat.JSON_OBJECT
            
        if config.stream:
            response = client.ask(
                args.question,
                args.system,
                response_format=response_format,
                prompt_template=config.prompt_template,
                tone=config.tone,
                output_format=config.output_format,
                stream=True,
            )
        else:
            response = client.ask(
                args.question,
                args.system,
                response_format=response_format,
                prompt_template=config.prompt_template,
                tone=config.tone,
                output_format=config.output_format,
            )
            if isinstance(response, dict):
                print(f"🤖 AI (JSON): {json.dumps(response, indent=2)}")
            else:
                print(f"🤖 AI: {response}")

        if client.config.last_prompt_file:
            print(f"🧩 Prompt file: {client.config.last_prompt_file}")
    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()
