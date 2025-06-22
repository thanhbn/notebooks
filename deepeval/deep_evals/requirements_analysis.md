# DeepEval Notebooks - Dependencies Analysis

## Comprehensive List of Required Packages

Based on analysis of all Jupyter notebooks in the `deep_evals` directory and its subdirectories, here are all the dependencies found:

### Core Dependencies

#### DeepEval & Evaluation
- `deepeval` - Core evaluation framework
- `pytest` - Testing framework integration
- `pytest-xdist` - Parallel test execution (optional)

#### Language Model & NLP
- `langchain` - LLM orchestration framework
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI integration (updated import path)
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `transformers` - Hugging Face transformers
- `sentencepiece` - Tokenization for transformers
- `accelerate` - Hugging Face acceleration

#### Data Processing & Analysis
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `datasets` - Hugging Face datasets library
- `json` - JSON data handling (built-in)

#### Text Analysis & Metrics
- `nltk` - Natural language toolkit
- `scikit-learn` - Machine learning utilities
- `textstat` - Text readability metrics
- `radon` - Code complexity analysis

#### Statistical Analysis
- `scipy` - Scientific computing
- `statsmodels` - Statistical modeling

#### Visualization
- `matplotlib` - Basic plotting
- `seaborn` - Statistical data visualization
- `plotly` - Interactive visualizations

#### Vector Stores & RAG
- `faiss-cpu` - Vector similarity search
- `chromadb` - Vector database
- `pinecone-client` - Pinecone vector database

#### Web & API
- `fastapi` - API framework
- `uvicorn` - ASGI server

#### Utilities
- `tqdm` - Progress bars
- `python-dotenv` - Environment variable management
- `jinja2` - Template engine
- `webbrowser` - Web browser control (built-in)
- `logging` - Logging functionality (built-in)
- `datetime` - Date and time handling (built-in)
- `time` - Time-related functions (built-in)
- `os` - Operating system interface (built-in)
- `sys` - System-specific parameters (built-in)
- `re` - Regular expressions (built-in)
- `ast` - Abstract syntax trees (built-in)
- `warnings` - Warning control (built-in)
- `typing` - Type hints (built-in)
- `dataclasses` - Data classes (built-in)
- `collections` - Container datatypes (built-in)
- `io` - Core I/O tools (built-in)
- `base64` - Base64 encoding (built-in)
- `shutil` - File operations (built-in)

#### Additional Phase-specific Dependencies
- `kaleido` - Static image export for plotly (optional)
- `ipython` - Interactive Python (for notebooks)

### Import Issues Found

Several notebooks have outdated imports that need to be updated:

1. **LangChain imports** - The import paths have changed:
   - OLD: `from langchain.embeddings import OpenAIEmbeddings`
   - NEW: `from langchain_openai import OpenAIEmbeddings`
   
   - OLD: `from langchain.chat_models import ChatOpenAI`
   - NEW: `from langchain_openai import ChatOpenAI`
   
   - OLD: `from langchain.vectorstores import FAISS`
   - NEW: `from langchain_community.vectorstores import FAISS`

2. **DeepEval imports** - Some imports may not be available:
   - `from deepeval import assert_test` - May need to check correct import path

### Recommended requirements.txt

```txt
# Core evaluation framework
deepeval>=0.20.0
pytest>=7.0.0
pytest-xdist>=3.0.0

# Language models and NLP
langchain>=0.1.0
langchain-community>=0.0.10
langchain-openai>=0.0.5
openai>=1.0.0
anthropic>=0.8.0
transformers>=4.30.0
sentencepiece>=0.1.99
accelerate>=0.20.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.10.0

# Text analysis
nltk>=3.8.0
scikit-learn>=1.3.0
textstat>=0.7.3
radon>=6.0.0

# Statistical analysis
scipy>=1.10.0
statsmodels>=0.14.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
kaleido>=0.2.1  # Optional for plotly static export

# Vector stores
faiss-cpu>=1.7.4
chromadb>=0.4.0
pinecone-client>=2.2.0

# Web and API
fastapi>=0.100.0
uvicorn>=0.23.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
jinja2>=3.1.0
ipython>=8.12.0
```

### Installation Commands

For a complete setup, run:

```bash
# Basic installation
pip install -r requirements.txt

# Additional NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"

# For development/testing
pip install jupyter notebook ipykernel
```

### Environment Variables Required

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Notes

1. Some notebooks contain mock/demo code that references functions like `get_llm_response()` which would need to be implemented.
2. The LangGraph integration mentioned in notebook 04 would require additional setup.
3. Some notebooks reference data files that should be present:
   - `data/rag_document.txt`
   - `data/code_samples.json`
4. Phase notebooks (Phase1-4) include comprehensive evaluation frameworks with advanced statistical analysis.