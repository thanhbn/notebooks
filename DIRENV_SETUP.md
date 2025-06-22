# Direnv Environment Setup

This repository uses [direnv](https://direnv.net/) to automatically manage virtual environments for different directories, ensuring each part of the project has the appropriate dependencies without conflicts.

## 🎯 Environment Structure

| Directory | Environment | Purpose | Virtual Env |
|-----------|-------------|---------|-------------|
| `/` (root) | **Root** | Main LangChain RAG tutorials | `venv_root` |
| `/langchain/` | **LangChain** | Comprehensive LangChain learning | `venv_langchain` |
| `/deepeval/deepeval_claude_created/` | **DeepEval** | AI model evaluation | `venv_deepeval` |
| `/Paper/1.3.3/` | **COTTON** | ML/PyTorch code generation | `venv_cotton` |
| `/AI-Papers/` | **AI-Papers** | PDF processing utilities | `venv_ai_papers` |
| `/ETL/`, `/langfuse/`, `/langgraph/` | **Shared Graph** | Graph-based applications | `venv_shared_graph` |

## 🚀 Quick Setup

### Prerequisites

1. **Install direnv:**
   ```bash
   # Ubuntu/Debian
   sudo apt install direnv
   
   # macOS
   brew install direnv
   
   # Arch Linux
   sudo pacman -S direnv
   ```

2. **Hook direnv to your shell:**
   ```bash
   # For bash users
   echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
   
   # For zsh users  
   echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
   
   # For fish users
   echo 'direnv hook fish | source' >> ~/.config/fish/config.fish
   ```

3. **Restart your shell or source your config file**

### Automated Setup

Run the setup script to create all environments:

```bash
./setup_direnv.sh
```

This script will:
- Create all virtual environments
- Install requirements for each environment
- Configure direnv for each directory

## 📁 Manual Environment Details

### 1. Root Environment (`/`)
**Purpose:** Main LangChain RAG development and `main.py` execution
**Key Dependencies:** Basic LangChain, OpenAI, Anthropic, ChromaDB

```bash
cd /mnt/d/llm/notebooks
# Environment auto-activates
python main.py
jupyter notebook langchain_rag_comprehensive_tutorial.ipynb
```

### 2. LangChain Environment (`/langchain/`)
**Purpose:** Comprehensive LangChain learning with all integrations
**Key Dependencies:** Full LangChain ecosystem, multiple vector DBs, evaluation tools

```bash
cd /mnt/d/llm/notebooks/langchain
# Environment auto-activates with enhanced LangChain setup
jupyter notebook 01_Models/01_ChatModels_Claude.ipynb
```

### 3. DeepEval Environment (`/deepeval/deepeval_claude_created/`)
**Purpose:** AI model evaluation and benchmarking
**Key Dependencies:** DeepEval, evaluation metrics, statistical analysis tools

```bash
cd /mnt/d/llm/notebooks/deepeval/deepeval_claude_created
# Environment auto-activates with evaluation focus
jupyter notebook 01_Foundation_and_Core_Concepts.ipynb
```

### 4. COTTON Environment (`/Paper/1.3.3/`)
**Purpose:** Machine learning research on Chain-of-Thought code generation
**Key Dependencies:** PyTorch, Transformers, PEFT, evaluation frameworks

```bash
cd /mnt/d/llm/notebooks/Paper/1.3.3
# Environment auto-activates with ML/PyTorch setup
python cotton_implementation.py
```

### 5. AI-Papers Environment (`/AI-Papers/`)
**Purpose:** PDF processing and text extraction utilities
**Key Dependencies:** PDF processing libraries (PyPDF, PyMuPDF, etc.)

```bash
cd /mnt/d/llm/notebooks/AI-Papers
# Environment auto-activates with PDF processing tools
python pdf_to_txt_converter.py
```

### 6. Shared Graph Environment (`/ETL/`, `/langfuse/`, `/langgraph/`)
**Purpose:** Graph-based AI applications, observability, and ETL workflows
**Key Dependencies:** LangGraph, LangFuse, advanced LangChain features

```bash
cd /mnt/d/llm/notebooks/ETL
# Environment auto-activates with graph capabilities
jupyter notebook langgraph_adv_part1_code_ingestion.ipynb

cd /mnt/d/llm/notebooks/langfuse
# Same shared environment
jupyter notebook langfuse_part1_installation.ipynb

cd /mnt/d/llm/notebooks/langgraph
# Same shared environment  
jupyter notebook langgraph_tutorial.ipynb
```

## 🔧 Customization

### Adding New Dependencies

1. **Update requirements.txt** in the relevant directory
2. **Reload the environment:**
   ```bash
   cd <directory>
   pip install -r requirements.txt
   ```

### Adding Environment Variables

Edit the `.envrc` file in any directory to add custom environment variables:

```bash
# Example: Add API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Reload
direnv reload
```

### Creating New Environments

1. **Create a new .envrc file:**
   ```bash
   # Check if virtual environment exists
   if [[ ! -d "venv_myproject" ]]; then
       python3 -m venv venv_myproject
   fi
   
   source venv_myproject/bin/activate
   export ENV_NAME="myproject"
   ```

2. **Allow the environment:**
   ```bash
   direnv allow
   ```

## 🛠️ Troubleshooting

### Environment Not Activating
```bash
# Check direnv status
direnv status

# Allow the environment
direnv allow

# Reload configuration
direnv reload
```

### Permission Denied
```bash
# Make sure you're in the right directory
pwd

# Check .envrc permissions
ls -la .envrc

# Allow direnv for the directory
direnv allow
```

### Dependencies Not Found
```bash
# Check which virtual environment is active
which python
echo $VIRTUAL_ENV

# Reinstall requirements
pip install -r requirements.txt

# For shared environments
pip install -r requirements_shared_graph.txt
```

### Clean Restart
```bash
# Remove all virtual environments
rm -rf venv_*

# Run setup script again
./setup_direnv.sh
```

## 📊 Environment Variables Reference

Each environment sets specific variables for optimal development:

| Variable | Purpose | Example |
|----------|---------|---------|
| `ENV_NAME` | Environment identifier | `"langchain"` |
| `PROJECT_ROOT` | Root directory path | `"/mnt/d/llm/notebooks"` |
| `PYTHONPATH` | Python module search path | Includes current and parent dirs |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer warnings | `false` |
| `HF_HOME` | Hugging Face cache location | `".cache/huggingface"` |
| `LANGCHAIN_TRACING_V2` | Enable LangChain tracing | `true` (LangChain env) |
| `DEEPEVAL_RESULTS_FOLDER` | Evaluation results path | `"./results"` (DeepEval env) |
| `TORCH_HOME` | PyTorch cache location | `".cache/torch"` (COTTON env) |

## 💡 Best Practices

1. **Always check environment before installing packages:**
   ```bash
   echo $ENV_NAME
   which python
   ```

2. **Use relative paths in notebooks** to maintain portability

3. **Keep requirements.txt files updated** when adding dependencies

4. **Test environment isolation** by checking package versions:
   ```bash
   pip list | grep langchain
   ```

5. **Use environment-specific cache directories** to avoid conflicts

6. **Regular cleanup** of unused virtual environments and cache files

## 🔄 Workflow Examples

### Switching Between Projects
```bash
# Work on LangChain tutorials
cd langchain
# Auto-activates venv_langchain with full LangChain ecosystem

# Switch to evaluation work  
cd ../deepeval/deepeval_claude_created
# Auto-activates venv_deepeval with evaluation tools

# Go back to main development
cd ../..
# Auto-activates venv_root with basic setup
```

### Running Different Notebook Types
```bash
# ETL workflow
cd ETL && jupyter notebook langgraph_adv_part1_code_ingestion.ipynb

# Evaluation workflow
cd deepeval/deepeval_claude_created && jupyter notebook 01_Foundation_and_Core_Concepts.ipynb

# Learning workflow
cd langchain && jupyter notebook 01_Models/01_ChatModels_Claude.ipynb
```

This setup ensures clean separation of concerns while maintaining easy navigation between different parts of the project. 🎯