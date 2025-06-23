# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-module educational LangChain/AI research repository focused on teaching Retrieval-Augmented Generation (RAG) systems and paper implementations. Contains learning materials (Vietnamese) and practical implementations across specialized domains.

## Repository Architecture

### Environment Management
Uses **direnv** for automatic environment isolation per module:
- Each directory has isolated dependencies via `.envrc` files
- Automatic virtual environment activation when entering directories
- Jupyter kernels registered for each environment

### Module Structure
1. **Root** (`/`): Core LangChain RAG tutorials and main notebooks
2. **LangChain** (`/langchain/`): Complete LangChain learning path with 8 modules
3. **DeepEval** (`/deepeval/`): AI evaluation framework with benchmarking
4. **LangGraph** (`/langgraph/`, `/ETL/`, `/langfuse/`): Graph-based workflows and monitoring
5. **Paper Implementations** (`/Paper/`): Research paper code implementations
6. **AI Papers** (`/AI-Papers/`): PDF processing and research paper analysis

## Development Commands

### Environment Setup
```bash
# Setup all environments with direnv (recommended)
./setup_simple.sh

# Register all Jupyter kernels
./register_kernels.sh

# Configure automatic kernel selection
python create_notebook_config.py
```

### Jupyter Development
```bash
# Start Jupyter Lab (recommended - supports multiple kernels)
jupyter lab

# Start basic notebook server
jupyter notebook

# Execute specific notebook
jupyter nbconvert --to notebook --execute <notebook_name>.ipynb

# List available kernels
jupyter kernelspec list
```

### Testing & Quality (DeepEval module)
```bash
# Navigate to deepeval directory first
cd deepeval/deep_evals/

# Run evaluation tests
pytest

# Run tests in parallel
pytest -n auto

# Run with coverage
pytest --cov=.
```

### Code Quality (LangChain module)
```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### API Development
```bash
# Development server with auto-reload
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production deployment
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Available Jupyter Kernels

The repository has specialized kernels for different domains:
- **🚀 Root (LangChain RAG)**: Main tutorials and comprehensive examples
- **🦜 LangChain (Complete)**: Full LangChain ecosystem with all integrations
- **📊 DeepEval (Evaluation)**: AI evaluation, benchmarking, and testing tools
- **🧠 COTTON (ML/PyTorch)**: Paper implementations with PyTorch/ML focus
- **📄 AI-Papers (PDF Utils)**: PDF processing and research analysis
- **🕸️ Graph (LangGraph/LangFuse)**: Graph workflows and observability

## Key Architecture Patterns

### RAG Pipeline Components
1. **Document Processing**: Multi-format loaders (PDF, HTML, CSV, JSON, web)
2. **Text Splitting**: Recursive and semantic chunking strategies  
3. **Vector Storage**: ChromaDB (dev), FAISS (performance), Pinecone (production)
4. **LLM Integration**: OpenAI, Anthropic, local models via Ollama
5. **Retrieval**: Similarity search, MMR, re-ranking strategies
6. **Monitoring**: LangSmith, LangFuse, DeepEval for evaluation

### Multi-Agent Systems (LangGraph)
- State management and graph-based workflows
- Agent orchestration and tool integration
- Async processing and error handling
- Visual debugging and monitoring

### Evaluation Framework (DeepEval)
- Automated testing for RAG systems
- Benchmark comparisons and metrics
- Performance monitoring and regression testing

## Key Dependencies by Module

**Core LangChain**: `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-community`, `langchain-core`
**Vector DBs**: `chromadb`, `pinecone-client`, `faiss-cpu`, `weaviate-client`
**Document Processing**: `pypdf`, `pymupdf`, `beautifulsoup4`, `unstructured`, `pdfplumber`
**LLM Providers**: `openai`, `anthropic`, `cohere`, `transformers`
**Evaluation**: `deepeval`, `ragas`, `langsmith`
**API/Web**: `fastapi`, `uvicorn`, `streamlit`, `gradio`
**Graph/Workflow**: `langgraph`, `networkx`, `langfuse`
**ML/Data**: `torch`, `transformers`, `datasets`, `pandas`, `numpy`