# Deep Query 

A powerful **Retrieval-Augmented Generation (RAG)** application with an advanced **Deep Research Agentic System** for comprehensive web research, document Q&A, and intelligent chat capabilities.

## ✨ Features

### 🔬 **Deep Research Agent**
- **Multi-Agent Architecture**: Hierarchical supervisor-worker pattern using LangGraph
- **Web Research**: Automated research using Tavily API and Playwright browser toolkit
- **Iterative Refinement**: Conducts up to 3 research cycles with strategic planning
- **Quality Control**: Built-in reflection tools and automatic scope management
- **Comprehensive Reports**: Generates detailed Markdown reports with inline citations

### 📚 **RAG (Retrieval-Augmented Generation)**
- **Document Upload**: Support for PDF, TXT, DOC, DOCX files
- **Vector Search**: Powered by Qdrant vector database
- **Smart Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Multi-backend Support**: OpenAI, Cohere, Google GenAI embeddings

### 💬 **Intelligent Chat**
- **Three Modes**:
  - **Chat Mode**: General conversation without documents
  - **RAG Mode**: Q&A with uploaded documents
  - **Deep Research Mode**: Comprehensive web-based research
- **Markdown Rendering**: Beautiful formatted responses with syntax highlighting
- **Code Block Support**: Proper rendering of code snippets with copy functionality

### 🎨 **Modern UI**
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Dark Mode**: Toggle between light and dark themes
- **Loading Animations**: Visual feedback for research progress
- **File Upload**: Drag-and-drop file upload with progress tracking

## 🏗️ Architecture

### Deep Research System
```
User Query → Supervisor Agent → Research Tasks
              ↓
         [Researcher 1] [Researcher 2] [Researcher 3]
              ↓              ↓              ↓
         Tavily Search | Browser Tools | Think Tool
              ↓
         Compression Layer (Organize + Citations)
              ↓
         Report Generator → Markdown Report
```

**Key Components**:
- **Supervisor Agent**: Strategic coordinator, breaks down queries, manages iterations
- **Researcher Agents**: Parallel workers executing web searches and browser automation
- **Compression Layer**: Organizes findings with citations while preserving all information
- **Report Generator**: Synthesizes research into comprehensive Markdown reports

### Tech Stack
- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Vanilla JavaScript with modern CSS
- **Database**: PostgreSQL with pgvector extension
- **Vector DB**: Qdrant
- **LLM Framework**: LangChain 0.3.27, LangGraph 0.6.7
- **Web Research**: Tavily API, Playwright
- **Monitoring**: Prometheus + Grafana

## 📋 Requirements

```
fastapi==0.116.1
uvicorn[standard]==0.35.0
python-multipart==0.0.20
python-dotenv==1.1.1
pydantic-settings==2.10.1
aiofiles==24.1.0
langchain==0.3.27
langchain-openai==0.3.34
langchain-community==0.3.29
langgraph==0.6.7
PyMuPDF==1.26.4
motor==3.7.1
openai==1.107.0
cohere==5.17.0
google-genai==1.35.0
qdrant-client==1.15.1
tavily-python==0.3.3
playwright==1.55.0
sqlalchemy==2.0.36
asyncpg==0.30.0
alembic==1.14.0
```

## 🚀 Installation

### Prerequisites

```bash
sudo apt update
sudo apt install libpq-dev gcc python3-dev
```

### 1. Clone the Repository

```bash
git clone https://github.com/Yousif-Abuzeid/RAG-APP.git
cd RAG-APP
```

### 2. Setup Environment Variables

#### Application Environment
```bash
cd docker/env
cp .env.app.example .env.app
```

Edit `.env.app` and configure:

```bash
# LLM Configuration
GENERATION_BACKEND="OPENAI"  # or GOOGLE_GENAI, COHERE
OPENAI_API_KEY="your-openai-key"
OPENAI_API_URL="https://api.openai.com/v1"  # or Ollama: http://172.17.0.1:11434/v1
GENERATION_MODEL_ID="gpt-4"  # or mistral:latest for Ollama

# Embeddings
EMBEDDING_BACKEND="GOOGLE_GENAI"
GOOGLE_GENAI_API_KEY="your-google-api-key"
EMBEDDING_MODEL_ID="models/text-embedding-004"

# Deep Research
TAVILY_API_KEY="tvly-your-tavily-api-key"
MAX_CONCURRENT_RESEARCH_UNITS=3  # Number of parallel research tasks per iteration
MAX_RESEARCHER_ITERATIONS=3      # Maximum number of research cycles

# Database
POSTGRES_USERNAME="minirag"
POSTGRES_PASSWORD="minirag"
POSTGRES_HOST="postgres"
POSTGRES_PORT=5432
POSTGRES_MAIN_DATABASE="minirag"

# Vector DB
VECTOR_DB_BACKEND="QDRANT"
QDRANT_HOST="qdrant"
QDRANT_PORT=6333
```

#### Database Environment
```bash
cd docker/env
cp env.postgres.example .env.postgres
```

### 3. Run with Docker Compose

```bash
cd docker
docker-compose up --build
```

**Services**:
- **FastAPI**: http://localhost:8000
- **Qdrant**: http://localhost:6333
- **PostgreSQL**: localhost:5432
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

### 4. Access the Application

Open your browser and navigate to: **http://localhost:8000**

## 🎯 Usage

### 1. **General Chat**
Just type your question and press send - no document upload needed!

### 2. **Document Q&A (RAG)**
1. Click the upload button (📎)
2. Select your document (PDF, TXT, DOC, DOCX)
3. Wait for processing and indexing
4. Ask questions about your document

### 3. **Deep Research** 🔬
1. Type your research question
2. Click the purple **Deep Research** button (🔍)
3. Watch the animated progress indicator showing:
   - 📋 Planning research tasks
   - 🔍 Conducting web research
   - 📊 Analyzing findings
   - 📝 Generating comprehensive report
4. Receive a detailed Markdown report with citations

**Example Research Queries**:
- "What are the latest developments in quantum computing?"
- "Compare React vs Vue.js for building web applications"
- "Explain the impact of AI on healthcare in 2024"

## 🔧 Configuration

### LLM Backends

#### Using OpenAI
```bash
GENERATION_BACKEND="OPENAI"
OPENAI_API_KEY="sk-..."
OPENAI_API_URL="https://api.openai.com/v1"
GENERATION_MODEL_ID="gpt-4"
```

#### Using Ollama (Local)
```bash
GENERATION_BACKEND="OPENAI"
OPENAI_API_KEY="ollama"
OPENAI_API_URL="http://172.17.0.1:11434/v1"
GENERATION_MODEL_ID="mistral:latest"
```

**Note**: For Ollama, ensure it's running on the host and listening on `0.0.0.0:11434`

#### Using Google GenAI
```bash
GENERATION_BACKEND="GOOGLE_GENAI"
GOOGLE_GENAI_API_KEY="AIza..."
GENERATION_MODEL_ID="gemini-1.5-pro"
```

### Deep Research Parameters

Configure research behavior via environment variables in `.env.app`:

```bash
# Deep Research Configuration
MAX_CONCURRENT_RESEARCH_UNITS=3  # Number of parallel research tasks per iteration
MAX_RESEARCHER_ITERATIONS=3      # Maximum number of research cycles
```

**Parameters**:
- `MAX_CONCURRENT_RESEARCH_UNITS`: Controls how many research agents work in parallel (default: 3)
- `MAX_RESEARCHER_ITERATIONS`: Maximum number of research-refinement cycles (default: 3)

These can also be overridden programmatically when initializing the DeepResearch class:

```python
from agents.deep_researcher import DeepResearch

researcher = DeepResearch(
    generation_client=client,
    max_concurrent_research_units=5,  # Override config
    max_researcher_iterations=4       # Override config
)
```

## 📊 API Endpoints

### Chat & Research
- `POST /api/v1/nlp/index/chat/{project_id}` - General chat
- `POST /api/v1/nlp/index/answer/{project_id}` - RAG Q&A
- `POST /api/v1/nlp/index/deep-research/{project_id}` - Deep research

### Document Management
- `POST /api/v1/data/upload/{project_id}` - Upload document
- `POST /api/v1/data/process/{project_id}` - Process document
- `POST /api/v1/nlp/index/push/{project_id}` - Index to vector DB

### Search
- `POST /api/v1/nlp/index/search/{project_id}` - Vector similarity search
- `GET /api/v1/nlp/index/info/{project_id}` - Collection info

## 🛠️ Development

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r src/requirments.txt

# Install Playwright browsers
playwright install --with-deps chromium

# Setup PostgreSQL and Qdrant locally
# Update connection strings in .env

# Run the application
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
cd src
pytest tests/
```

## 🐛 Troubleshooting

### Deep Research Returns Network Error
**Solution**: The CORS middleware should be enabled. If using Ollama, ensure:
1. Ollama is listening on `0.0.0.0:11434` (not just `127.0.0.1`)
2. Docker bridge network is accessible via `172.17.0.1`

### Model Doesn't Support Tool Calling
**Solution**: Use models with tool calling support:
- ✅ mistral:latest, mixtral, llama3.1
- ✅ GPT-4, GPT-3.5-turbo
- ✅ Gemini 1.5 Pro
- ❌ gemma, phi (no tool support)

### Playwright Browser Not Found
**Solution**: 
```bash
docker exec -it <container-name> playwright install --with-deps chromium
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Tavily](https://tavily.com/) - Web search API
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ using LangChain, LangGraph, and FastAPI
