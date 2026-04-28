# Azure AI Search RAG System

Enterprise-ready Retrieval-Augmented Generation (RAG) system powered by Azure AI Search for production-scale document search and question answering.

## 🎯 Overview

This system provides intelligent document search and question answering using Azure's managed search infrastructure combined with OpenAI's language models. It supports hybrid search (keyword + semantic) for optimal relevance and can handle enterprise-scale document collections.

## 🏗️ Architecture

```
User Query → Azure AI Search (Hybrid Search) → Document Retrieval → Azure OpenAI → Final Answer
```

### Core Components
- **Azure AI Search**: Enterprise search service with vector + text capabilities
- **Azure OpenAI**: GPT-4.1-nano for answer generation
- **OpenAI Embeddings**: text-embedding-3-small for semantic search
- **Hybrid Search Engine**: Combines BM25 (keyword) + vector similarity

## ⚡ Quick Start

### Prerequisites
- Azure subscription with AI Search and OpenAI services
- Python 3.8+ with required packages
- Document collection indexed in Azure AI Search

### Installation
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with your Azure credentials:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY="your-azure-openai-key"
AZURE_OPENAI_ENDPOINT="https://your-service.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4.1-nano"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
AZURE_SEARCH_QUERY_KEY="your-query-key"
AZURE_SEARCH_ADMIN_KEY="your-admin-key"  # fallback
```

## 🚀 Usage

### Command Line Mode
```bash
# Ask a single question
python apps/rag/azure_search_rag.py --query "What types of insurance policies are available?"

# Configure result count
python apps/rag/azure_search_rag.py --query "How do I file a claim?" --top-k 5

# Use pure vector search
python apps/rag/azure_search_rag.py --query "coverage options" --no-hybrid
```

### Interactive Mode
```bash
python apps/rag/azure_search_rag.py
```
```
🤖 Azure AI Search RAG - Interactive Mode
💡 Type 'quit' to exit
--------------------------------------------------

🤔 Ask anything: What should I do if my car is stolen?
🔍 Searching Azure AI Search...
🤖 Answer: If your car is stolen and you have insurance, immediately 
          report it to police within 24 hours and contact your insurance 
          company to file a claim under comprehensive coverage...
```

### Programmatic Usage
```python
from apps.rag.azure_search_rag import AzureSearchRAG

# Initialize the RAG system
rag = AzureSearchRAG()

# Ask questions
answer = rag.ask("What is the claims process?", top_k=3)
print(answer)

# Direct search for debugging
results = rag.search("insurance policy", top_k=5)
for result in results:
    print(f"Source: {result['source']}")
    print(f"Content: {result['content'][:200]}...")
```

## 🔍 Search Capabilities

### Hybrid Search (Default)
Combines two powerful search algorithms:
- **Keyword Search (BM25)**: Finds exact term matches, great for specific terminology
- **Vector Search**: Semantic similarity using embeddings, great for conceptual matching

### Search Modes
```python
# Hybrid search (recommended)
results = rag.search("car insurance claim", use_hybrid=True)

# Pure vector search
results = rag.search("vehicle coverage", use_hybrid=False)
```

### Field Selection
Returns essential document fields:
- `content`: Document text chunk
- `source`: File path for attribution
- `title`: Document title
- `chunkIndex`: Position within larger document

## 📊 Performance & Scalability

### Response Times
- **Query Processing**: Sub-second responses
- **Embedding Generation**: ~100-200ms
- **Azure Search**: ~50-100ms
- **LLM Generation**: ~500ms-2s

### Scale Characteristics
- **Document Capacity**: Millions of documents supported
- **Concurrent Users**: Auto-scaling Azure infrastructure
- **Index Size**: Configurable search units
- **Global Deployment**: Multi-region support available

## 🛡️ Enterprise Features

### Authentication & Security
- **Robust Key Management**: Automatic fallback between query/admin keys
- **Azure AD Integration**: Enterprise identity management
- **Network Security**: VNet and private endpoint support
- **Compliance**: SOC, HIPAA, ISO certifications

### Monitoring & Observability
- **Azure Monitor Integration**: Search analytics and performance metrics
- **Custom Logging**: Debug information for troubleshooting
- **Error Handling**: Graceful degradation and retry logic
- **Health Checks**: Index validation and connectivity testing

## 📁 Project Structure

```
apps/rag/
├── azure_search_rag.py      # Main RAG system
├── embeddings.py            # Embedding generation service
└── mini_rag.py              # Lightweight alternative

scripts/
├── ingest.py                # Document ingestion utilities
└── azure_search_migrator.py # Migration tools

infra/
├── azure-openai-setup.bicep # Azure resource deployment
└── azure-openai-setup.json  # ARM template parameters
```

## 🔧 Configuration Options

### Search Parameters
- `top_k`: Number of documents to retrieve (default: 3)
- `use_hybrid`: Enable/disable hybrid search (default: True)
- `similarity_threshold`: Minimum relevance score (optional)

### Index Configuration
- **Index Name**: `enterprise-rag-index`
- **Vector Field**: `contentVector`
- **Embedding Model**: `text-embedding-3-small`
- **Vector Dimensions**: 1536

### LLM Configuration
- **Model**: `gpt-4.1-nano`
- **Max Tokens**: 1000 (configurable)
- **Temperature**: 0.7 (configurable)
- **Response Format**: Plain text with source attribution

## 🚨 Troubleshooting

### Common Issues

**Authentication Errors**
```
❌ The given API key doesn't match service's internal keys
```
- Verify `AZURE_SEARCH_QUERY_KEY` and `AZURE_SEARCH_ADMIN_KEY`
- Check service endpoint URL format
- Ensure keys have proper permissions

**No Results Returned**
- Verify index contains documents: Check Azure portal
- Test with simpler queries: Try single keywords
- Check embedding model compatibility: Must match index

**Slow Response Times**
- Monitor Azure Search metrics in portal 
- Consider increasing search units for higher load
- Optimize query complexity and result count

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

rag = AzureSearchRAG()
```

## 📈 Production Deployment

### Azure Infrastructure
1. **Azure AI Search**: Standard tier for production workloads
2. **Azure OpenAI**: Deployed in same region for latency
3. **Azure Monitor**: Logging and alerting configuration
4. **Azure Key Vault**: Secure credential management

### Performance Optimization
- **Search Units**: Scale based on query volume and index size
- **Caching**: Implement query result caching for frequent questions
- **Load Balancing**: Multiple search service instances for high availability
- **Geographic Distribution**: Multi-region deployment for global users

### Monitoring Setup
```bash
# Azure CLI commands for monitoring configuration
az monitor log-analytics workspace create --resource-group myRG --workspace-name myWorkspace
az monitor diagnostic-settings create --resource mySearchService --logs '[{"category":"OperationLogs","enabled":true}]'
```

## 🤝 Contributing

This RAG system is designed for enterprise use. When contributing:
1. Maintain backward compatibility with existing integrations
2. Include comprehensive error handling and logging
3. Update documentation for any API changes
4. Test with realistic document collections and query volumes

## 📄 License

[MIT License](LICENSE) - Enterprise-friendly open source licensing

## 🆘 Support

For enterprise support and custom implementations:
- Review Azure AI Search documentation
- Check Azure OpenAI service limits and quotas
- Monitor Azure status page for service health
- Consider Azure professional services for large deployments

---

**Built for Production** | **Azure Native** | **Enterprise Ready** | **Hybrid Search Powered**