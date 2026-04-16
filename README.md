# Enterprise AI Assistant - LLM Client

<!-- This README focuses on local CLI usage, provider setup, and the reusable prompt workflow. -->

A simple, flexible CLI tool to ask anything and get AI responses. Supports multiple AI providers including OpenAI, Azure OpenAI (Microsoft Foundry), and GitHub Models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
root folder
pip install -r requirements.txt
```

### 2. Quick Demo - Enhanced Search System
```bash
# Test the enhanced RAG search system
cd apps\rag
python improved_search.py --query "machine learning" --hybrid --rerank --boost-recent --show-explanation

# Or try interactive mode
python improved_search.py --interactive
```

### 3. Set Up API Keys

#### Option A: OpenAI (Default)
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

#### Option B: Azure OpenAI (Microsoft Foundry)

Step 1. Run the Bicep deployment from the repository root.

```powershell
az deployment group create --resource-group rg-enterprise-ai-assistant --template-file .\infra\azure-openai-setup.bicep --parameters accountName=ai-enterprise-ai-assistant-dev01
```
<!-- # new accountName each time -->

The template now creates both a chat deployment and an embeddings deployment by default. The daily cleanup Logic App is disabled by default. Enable it only when you want the environment deleted automatically:

```powershell
az deployment group create --resource-group rg-enterprise-ai-assistant --template-file .\infra\azure-openai-setup.bicep --parameters accountName=ai-enterprise-ai-assistant-dev01 deployDailyDeleteSchedule=true
```
```powershell to check embeddings created
python embeddings.py --provider azure --text "Explain vector search" --summary
```

Step 2. Set the environment variables in PowerShell.

```powershell
$env:AI_PROVIDER="azure"
$env:AZURE_OPENAI_API_KEY="your-azure-key"
$env:AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
# Chat deployment used by openai_client.py:
$env:AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4.1-nano"
# Embeddings deployment used by ../rag/embeddings.py:
$env:AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
# Optional: print an embedding summary after each chat turn:
# $env:AI_SHOW_EMBEDDINGS="true"
# $env:AI_EMBEDDING_PREVIEW_LENGTH="5"
# Optional explicit v1 base URL override:
# $env:AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/"
```

#### Option C: GitHub Models (Free to start)
```bash
export AI_PROVIDER="github"
export GITHUB_TOKEN="your-github-token"
export GITHUB_MODEL="gpt-4o-mini"
```

### 4. Run the Tool

#### Interactive Mode (Default)
```bash
python .\apps\llm\openai_client.py
```

#### CLI Chatbot With History
```bash
python openai_client.py -p azure
python openai_client.py -p azure --show-embeddings
```

The interactive CLI now behaves like a chatbot: each new message includes prior turns, and you can reset or inspect history at any time.
Conversation state is also saved to `apps/llm/.chat_state.json` by default, so restarting the CLI continues the previous chat unless you clear it.
Before each `Ask anything` prompt, the CLI asks whether to begin a new chat session. Choosing `y` clears the previous saved conversation and starts fresh; choosing `n` continues the current conversation.
When embedding previews are enabled, the CLI prints a compact embedding summary for each completed assistant response using the configured embeddings deployment.

#### Single Question Mode
```bash
python openai_client.py -q "What is artificial intelligence?"
```

#### Structured JSON Response
```bash
python openai_client.py --json -q "List 3 programming languages as JSON"
```

#### Streaming Response
```bash
python openai_client.py --stream -q "Tell me a story"
```

#### Controlled Tone And Format
```bash
python openai_client.py --tone technical --format bullet -q "Explain vector databases"
```

#### Embedding Generator

```bash
cd ../rag
python embeddings.py --provider azure --text "Explain vector search"
python embeddings.py --provider azure --text "Explain vector search" --summary
```

The embeddings CLI uses `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` by default for Azure, or `EMBEDDING_MODEL` for non-Azure runs.
Use `--summary` to print a compact inspection view instead of the full JSON payload. Add `--preview-length 10` to show more vector values.

#### Reusable Prompt Templates
```bash
python openai_client.py --template summarizer -q "Summarize this meeting transcript: ..."
python openai_client.py --template rag -q "Using the provided context, what are the SLA exceptions? Context: ..."
python openai_client.py --template json_extractor -q "Extract name, company, email, and region from: Jane Doe from Contoso can be reached at jane@contoso.com in EMEA"
python openai_client.py --template chatbot_personality -q "Draft a helpful response to a customer asking about onboarding"
python openai_client.py --template code_generator -q "Write a Python function that groups invoices by customer"
```

#### Automatic Prompt Routing
```bash
python openai_client.py -q "Summarize this transcript: ..."
python openai_client.py -q "Extract name, date, and amount from this invoice"
python openai_client.py -q "Write a Python function to validate email addresses"
```

The client loads prompt content from the `apps/llm/prompts/` folder, infers the intent of the question, and selects the matching prompt file automatically unless you override it with `--template`.

## 📋 Usage Examples

### Interactive Mode
```bash
$ python openai_client.py
✅ Connected to OPENAI (gpt-3.5-turbo)
🧠 Enterprise AI Assistant - Interactive Mode
💡 Type 'quit', 'exit', or 'q' to stop
💡 Type 'help' for commands
--------------------------------------------------

🤔 Ask anything: What is Python?
🤖 AI: Python is a high-level programming language...

🤔 Ask anything: json List 3 colors
🤖 AI (JSON): {
  "colors": ["red", "blue", "green"]
}

🤔 Ask anything: stream Tell me about AI
🤖 AI: Artificial Intelligence (AI) refers to...
[Response streams in real-time]

🤔 Ask anything: help
📋 Available commands:
  help       - Show this help
  models     - List available models
  config     - Show current configuration
  history    - Show the current chat history
  clear      - Clear chat history and delete saved state
  system <p> - Set a custom system prompt and reset chat history
  system clear - Remove the custom system prompt and reset chat history
  json <q>   - Ask question with JSON response
  stream <q> - Ask question with streaming response
  quit/q     - Exit the program
```

### Single Questions
```bash
# Basic question
python openai_client.py -q "Explain machine learning in simple terms"

# With system prompt
python openai_client.py -q "Write a Python function to sort a list" -s "You are a Python expert"

# Use the built-in code generator prompt
python openai_client.py --template code_generator -q "Write a FastAPI health check endpoint"

# Control tone and structure
python openai_client.py --tone teacher --format steps -q "Explain APIs to a beginner"

# Use the JSON extractor template for structured output
python openai_client.py --template json_extractor -q "Extract order_id, status, and total from: Order A-42 is shipped and totals $18.50"

# JSON structured response
python openai_client.py --json -q "List 5 benefits of AI as JSON with categories"

# Streaming response
python openai_client.py --stream -q "Write a detailed explanation of quantum computing"

# Different provider
python openai_client.py -p github -q "What are the benefits of AI?"

# Specific model with custom settings
python openai_client.py -m gpt-4 --temperature 0.9 -q "Write a creative poem about coding"
```

### Configuration
```bash
# Show current configuration
python openai_client.py --config

# Custom settings
python openai_client.py --max-tokens 500 --temperature 0.9 -q "Be creative!"

# Force automatic prompt selection explicitly
python openai_client.py --template auto -q "Summarize the following design notes"

# Start a chatbot with a custom system prompt
python openai_client.py -p azure -s "You are a concise project assistant"
```

## ⚙️ Configuration Options

### Command Line Arguments
- `-q, --question`: Ask a single question and exit
- `-p, --provider`: AI provider (openai, azure, github)
- `-m, --model`: Model to use
- `-s, --system`: System prompt
- `--template`: Reusable prompt template or `auto` intent routing (`auto`, `chatbot_personality`, `summarizer`, `rag`, `json_extractor`, `code_generator`)
- `--tone`: Tone preset (`balanced`, `concise`, `friendly`, `technical`, `executive`, `teacher`)
- `--format`: Output format preset (`paragraph`, `bullet`, `steps`, `summary`, `table`)
- `--max-tokens`: Maximum tokens in response
- `--temperature`: Response creativity (0.0-1.0)
- `--json`: Request JSON response format
- `--stream`: Enable streaming response
- `--config`: Show configuration and exit

### Environment Variables
- `AI_PROVIDER`: Provider (openai, azure, github)
- `OPENAI_API_KEY`: OpenAI API key
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Azure chat deployment name used by `openai_client.py`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Azure embeddings deployment name used by `../rag/embeddings.py`
- `EMBEDDING_MODEL`: Default embedding model or deployment name fallback
- `AZURE_OPENAI_BASE_URL`: Optional explicit Azure v1 endpoint
- `GITHUB_TOKEN`: GitHub personal access token
- `AI_PROMPT_TEMPLATE`: Default reusable prompt template or `auto`
- `AI_SYSTEM_PROMPT`: Default custom system prompt for the chatbot
- `AI_PERSIST_CHAT_HISTORY`: Save chat history between runs (`true` by default)
- `AI_CHAT_STATE_FILE`: Optional path to the saved chat state JSON file
- `AI_TONE`: Default tone preset
- `AI_OUTPUT_FORMAT`: Default output format preset
- `AI_MAX_TOKENS`: Maximum response tokens (default: 1000)
- `AI_TEMPERATURE`: Response creativity (0.0-1.0, default: 0.7)
- `AI_RESPONSE_FORMAT`: Default response format (text, json, json_schema)
- `AI_STREAM`: Enable streaming by default (true/false)

## 🎯 Features

<!-- Feature bullets below reflect the current CLI behavior and configuration surface. -->

✅ **Multi-Provider Support**
- OpenAI (GPT-3.5, GPT-4)
- Azure OpenAI (Microsoft Foundry)
- GitHub Models (Free tier available)

✅ **Enhanced RAG Search System**
- Advanced enhanced search with multi-factor ranking (semantic + keyword + recency + boost)
- Smart document chunking with Microsoft best practices (1500 chars, 300 overlap)
- Hybrid search combining TF-IDF and semantic similarity
- Re-ranking pipeline with configurable weights and diversity filtering
- Basic semantic search with FAISS vector similarity
- Interactive search modes with comprehensive analytics

✅ **Advanced Response Formats**
- Text responses (default)
- JSON object responses
- JSON schema validation support
- Streaming responses for real-time output

✅ **Flexible Usage**
- Interactive chat mode with enhanced commands
- Multi-turn CLI chatbot with preserved chat history across restarts by default
- Custom system prompt support from CLI flags, environment variables, or interactive commands
- Single question mode with format options
- System prompts support
- Reusable prompt files loaded from the `prompts/` folder
- Automatic intent routing to summarization, RAG, JSON extraction, chatbot, or code generation prompts
- Client configuration tracks the last prompt template and prompt file used
- Reusable tone presets and output format presets
- Real-time streaming output

✅ **Production RAG Capabilities**
- Document chunking with multiple strategies (fixed, sentence, semantic, hybrid)
- Enhanced metadata tracking (keywords, dates, relationships, boost factors)
- Multiple similarity metrics (cosine, dot product, euclidean)
- Comprehensive search statistics and performance monitoring
- Enterprise-grade JSON serialization and state persistence

✅ **Easy Configuration**
- Environment variables
- Command line arguments
- Built-in help system
- Interactive commands (json, stream)

✅ **Error Handling**
- Graceful error messages
- Connection validation
- JSON parsing validation
- Keyboard interrupt handling

## 🔧 Troubleshooting

### Common Issues

**"OpenAI library not found"**
```bash
pip install openai requests
```

**"Failed to initialize client"**
- Check your API key is set correctly
- Verify your internet connection
- For Azure: Ensure endpoint and model name are correct

**"Authentication failed"**
- Verify your API key is valid and active
- Check if you have sufficient credits/quota

### Getting API Keys

1. **OpenAI**: Visit [platform.openai.com](https://platform.openai.com/api-keys)
2. **Azure OpenAI**: Create resource in Azure portal
3. **GitHub**: Create token at [github.com/settings/tokens](https://github.com/settings/tokens)

## 🔍 Enhanced RAG Search System

The Enterprise AI Assistant features a comprehensive search system with both basic semantic search and advanced enhanced search capabilities for production-grade RAG (Retrieval Augmented Generation) applications.

### ✨ Enhanced Search (Recommended)

**Advanced search with improved ranking, chunking, and hybrid capabilities:**

```bash
cd apps\rag

# Quick start with sample data
python improved_search.py --build-with-chunks --index-path enhanced.faiss

# Enhanced search with all features
python improved_search.py --query "machine learning deployment" --hybrid --rerank --boost-recent --show-explanation

# Interactive enhanced search
python improved_search.py --interactive
```

#### ⭐ Key Enhanced Features

**🎯 Advanced Ranking System:**
- **Multi-Factor Scoring**: Semantic similarity + TF-IDF keywords + recency boost + manual boost factors
- **Hybrid Search**: Configurable blending of semantic (70%) and keyword (30%) search
- **Re-ranking Pipeline**: Multiple scoring factors applied for better relevance
- **Diversity Filtering**: Prevents similar results from dominating output

**🔪 Smart Document Chunking:**
- **Microsoft Best Practices**: Follows Azure Search recommendations (1500 chars, 300 overlap)
- **Multiple Strategies**: Fixed-size, sentence-aware, semantic (paragraph), hybrid chunking
- **Context Preservation**: Maintains document relationships and chunk metadata
- **Granular Search**: Search within document chunks for better precision

**📊 Production Features:**
- **Enhanced Metadata**: Keywords, creation dates, boost factors, chunk relationships
- **Comprehensive Analytics**: Detailed statistics and performance metrics
- **Flexible Configuration**: Multiple similarity metrics, configurable weights
- **Enterprise Ready**: JSON serialization, state persistence, monitoring capabilities

#### Enhanced Search Commands

```bash
# Basic enhanced search
python improved_search.py --query "artificial intelligence"

# Full enhanced search with explanations
python improved_search.py --query "deep learning" --hybrid --rerank --boost-recent --show-explanation --show-content

# Custom hybrid weights (80% semantic, 20% keyword)
python improved_search.py --query "AI ethics" --hybrid --alpha 0.8

# Different similarity metrics
python improved_search.py --query "neural networks" --similarity dot_product

# Interactive mode with advanced commands
python improved_search.py --interactive
# Interactive commands: <query>, hybrid <query>, plain <query>, stats, help, quit
```

#### Building Enhanced Indexes

```bash
# Sample documents with chunking (recommended)
python improved_search.py --build-with-chunks

# Custom documents from directory
python improved_search.py --build-from-dir /path/to/docs --index-path custom.faiss

# Configure chunking strategy
python improved_search.py --build-with-chunks --chunk-strategy sentence --chunk-size 1200 --overlap 200
```

### 🔍 Basic Semantic Search

**Simple vector-based document search using FAISS:**

```bash
# Build basic index
python semantic_search.py --build-sample --index-path enterprise.faiss

# Basic semantic search
python semantic_search.py --query "machine learning basics" --index-path enterprise.faiss

# Interactive basic search
python semantic_search.py --interactive --index-path enterprise.faiss
```

### 📊 Performance Comparison

**Enhanced vs Basic Search Results for "machine learning deployment":**

```
Enhanced Search (improved_search.py):
#1 📄 Large Language Model Deployment Strategies - Score: 0.7134
   🎯 Score Breakdown: Semantic: 0.645, Keyword: 0.089, Recency: 1.2x, Boost: 1.5x
   ✂️  Chunk: 0 (490 chars, strategy: fixed)
   🔑 Keywords: deployment, deploying, large, language, models

Basic Search (semantic_search.py):  
#1 📄 Large Language Model Deployment Strategies - Score: 0.6245
   🏷️  Full document search only
```

### 🛠️ Integration with Embeddings

Test the underlying embeddings system:
```bash
python embeddings.py --provider azure --text "machine learning algorithms" --summary
```

### 📈 Index Statistics

View comprehensive analytics:
```bash
# Enhanced search statistics
python improved_search.py --stats

# Basic search statistics  
python semantic_search.py --stats --index-path enterprise.faiss
```

### 🎯 Sample Data

Both search systems include high-quality sample documents covering:
- **Artificial Intelligence**: Fundamentals and 2024 developments
- **Machine Learning**: Advanced techniques and algorithms
- **Deep Learning**: Neural networks and current trends
- **LLM Deployment**: Production strategies and best practices
- **AI Ethics**: Responsible AI development principles
- **Vector Search**: Advanced embeddings and search techniques
- **RAG Systems**: Production best practices and implementation

### 🚀 Production Recommendations

**For Production RAG Systems:**
- ✅ Use `improved_search.py` for better ranking and chunking
- ✅ Enable hybrid search for keyword + semantic relevance
- ✅ Use chunking for better granularity and context windows
- ✅ Enable re-ranking for multi-factor scoring
- ✅ Monitor with statistics and scoring explanations

**For Simple Prototypes:**
- ✅ Use `semantic_search.py` for basic vector similarity
- ✅ Fast setup with minimal configuration
- ✅ Good for initial RAG experimentation

Both systems use Azure OpenAI's `text-embedding-3-small` model for 1536-dimensional embeddings with optimized FAISS IndexFlatIP for fast cosine similarity search.



Built with ❤️ for the Enterprise AI Assistant project
