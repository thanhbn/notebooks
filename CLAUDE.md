# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain educational and development repository focused on teaching Retrieval-Augmented Generation (RAG) systems. The project contains both learning materials (in Vietnamese) and practical code implementations.

## Key Files

- `langchain_learning_roadmap.md`: Comprehensive 12-week learning roadmap covering LangChain from basics to production
- `langchain_rag_comprehensive_tutorial.ipynb`: Jupyter notebook with practical RAG implementation examples
- `main.py`: Basic Python template file (minimal functionality)

## Development Commands

### Working with Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Execute notebook from command line
jupyter nbconvert --to notebook --execute langchain_rag_comprehensive_tutorial.ipynb
```

### API Development (when building FastAPI apps)
```bash
# Run FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production deployment
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture Notes

The project follows a progressive learning structure:
1. **Document Processing**: Uses loaders for PDF, HTML, and various formats
2. **Vector Stores**: Integrates with ChromaDB, Pinecone, and FAISS for embeddings
3. **LLM Integration**: Supports OpenAI, Anthropic, and other providers
4. **RAG Pipeline**: Implements retrieval, ranking, and generation components
5. **Production Patterns**: Includes monitoring with LangSmith and deployment strategies

## Key Dependencies

The project uses these main packages:
- `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-community`
- Vector DBs: `chromadb`, `pinecone-client`, `faiss-cpu`
- Document processing: `beautifulsoup4`, `pypdf`, `pymupdf`
- API/Web: `fastapi`, `uvicorn`
- Monitoring: `ragas`, `langsmith`