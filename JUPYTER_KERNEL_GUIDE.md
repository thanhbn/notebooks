# Jupyter Kernel Management Guide

This guide explains how Jupyter notebooks automatically detect and use the correct Python environment for each directory in your repository.

## 🧠 How It Works

### Automatic Kernel Registration
Each virtual environment is registered as a separate Jupyter kernel with a descriptive name and icon:

| Directory | Kernel Name | Display Name | Environment |
|-----------|-------------|-------------|-------------|
| `/` (root) | `root-env` | 🚀 Root (LangChain RAG) | `venv_root` |
| `/langchain/` | `langchain-env` | 🦜 LangChain (Complete) | `venv_langchain` |
| `/deepeval/deepeval_claude_created/` | `deepeval-env` | 📊 DeepEval (Evaluation) | `venv_deepeval` |
| `/Paper/1.3.3/` | `cotton-env` | 🧠 COTTON (ML/PyTorch) | `venv_cotton` |
| `/AI-Papers/` | `ai-papers-env` | 📄 AI-Papers (PDF Utils) | `venv_ai_papers` |
| `/ETL/`, `/langfuse/`, `/langgraph/` | `graph-env` | 🕸️ Graph (LangGraph/LangFuse) | `venv_shared_graph` |

### Environment Variables
Each `.envrc` file sets a `JUPYTER_DEFAULT_KERNEL` variable that indicates the preferred kernel for that directory.

## 📚 Usage Workflows

### 1. Starting Jupyter Lab/Notebook

**From any directory:**
```bash
# Navigate to your desired directory
cd /mnt/d/llm/notebooks/langchain

# Start Jupyter (environment auto-activated by direnv)
jupyter lab
# or
jupyter notebook
```

### 2. Creating New Notebooks

When you create a new notebook, Jupyter will:
1. **Show available kernels** in the kernel selection dialog
2. **Highlight the recommended kernel** for the current directory
3. **Auto-select the appropriate kernel** based on location

**Example workflow:**
```bash
cd langchain
jupyter lab
# → New notebook defaults to "🦜 LangChain (Complete)" kernel

cd ../deepeval/deepeval_claude_created  
jupyter lab
# → New notebook defaults to "📊 DeepEval (Evaluation)" kernel
```

### 3. Opening Existing Notebooks

**Automatic detection:**
- Existing notebooks remember their last used kernel
- If the kernel is unavailable, Jupyter will prompt you to select a new one
- The directory-appropriate kernel will be highlighted as recommended

**Manual kernel switching:**
- **Jupyter Lab:** `Kernel` menu → `Change Kernel...`
- **Jupyter Notebook:** `Kernel` menu → `Change kernel`

## 🔧 Advanced Usage

### Checking Available Kernels
```bash
# List all registered kernels
jupyter kernelspec list

# Expected output:
# Available kernels:
#   root-env         /home/user/.local/share/jupyter/kernels/root-env
#   langchain-env    /home/user/.local/share/jupyter/kernels/langchain-env
#   deepeval-env     /home/user/.local/share/jupyter/kernels/deepeval-env
#   cotton-env       /home/user/.local/share/jupyter/kernels/cotton-env
#   ai-papers-env    /home/user/.local/share/jupyter/kernels/ai-papers-env
#   graph-env        /home/user/.local/share/jupyter/kernels/graph-env
```

### Verifying Kernel Environment
Inside a notebook, check which environment you're using:

```python
# Check Python executable path
import sys
print("Python executable:", sys.executable)

# Check environment name
import os
print("Environment:", os.environ.get('ENV_NAME', 'unknown'))

# Check available packages
import pkg_resources
packages = [d.project_name for d in pkg_resources.working_set]
print("Key packages:", [p for p in packages if 'langchain' in p or 'deepeval' in p])
```

### Environment-Specific Features

#### LangChain Environment (`🦜 LangChain (Complete)`)
```python
# Verify LangChain ecosystem
import langchain
import langchain_anthropic
import langchain_openai
print("LangChain version:", langchain.__version__)

# Check tracing
import os
print("Tracing enabled:", os.environ.get('LANGCHAIN_TRACING_V2'))
```

#### DeepEval Environment (`📊 DeepEval (Evaluation)`)
```python
# Verify evaluation tools
import deepeval
from deepeval.metrics import AnswerRelevancyMetric
print("DeepEval version:", deepeval.__version__)

# Check results folder
import os
print("Results folder:", os.environ.get('DEEPEVAL_RESULTS_FOLDER'))
```

#### COTTON Environment (`🧠 COTTON (ML/PyTorch)`)
```python
# Verify PyTorch setup
import torch
import transformers
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers version:", transformers.__version__)
```

### Kernel Management Commands

**Re-register all kernels:**
```bash
./register_kernels.sh
```

**Remove a specific kernel:**
```bash
jupyter kernelspec remove <kernel-name>
# Example: jupyter kernelspec remove langchain-env
```

**Remove all custom kernels:**
```bash
jupyter kernelspec remove root-env langchain-env deepeval-env cotton-env ai-papers-env graph-env
```

## 🛠️ Troubleshooting

### Problem: Kernel Not Found
**Symptoms:** Notebook shows "Kernel not found" or fails to start

**Solution:**
```bash
# 1. Check if kernels are registered
jupyter kernelspec list

# 2. Re-register kernels if missing
./register_kernels.sh

# 3. Verify virtual environment exists
ls -la venv_*
```

### Problem: Wrong Environment Activated
**Symptoms:** Wrong packages available, import errors

**Solution:**
```python
# 1. Check current environment in notebook
import sys
print("Python path:", sys.executable)

# 2. Check available packages
import pkg_resources
installed = [d.project_name for d in pkg_resources.working_set]
print("LangChain packages:", [p for p in installed if 'langchain' in p])

# 3. Change kernel if needed
# Use Kernel → Change Kernel menu in Jupyter
```

### Problem: Kernel Crashes on Startup
**Symptoms:** Kernel dies immediately, connection errors

**Solution:**
```bash
# 1. Check virtual environment health
cd <directory>
source <venv>/bin/activate
python -c "import sys; print(sys.version)"

# 2. Reinstall ipykernel in the environment
pip install --force-reinstall ipykernel

# 3. Re-register the kernel
./register_kernels.sh
```

### Problem: Package Not Found
**Symptoms:** `ModuleNotFoundError` for expected packages

**Solution:**
```bash
# 1. Verify you're in the correct environment
echo $ENV_NAME
echo $JUPYTER_DEFAULT_KERNEL

# 2. Check if package is installed in the kernel's environment
cd <environment-directory>
source <venv>/bin/activate
pip list | grep <package-name>

# 3. Install missing package
pip install <package-name>

# 4. May need to restart kernel in Jupyter
```

## 📊 Kernel Selection Strategies

### For Development Work
- **LangChain learning:** Use `🦜 LangChain (Complete)`
- **Model evaluation:** Use `📊 DeepEval (Evaluation)`  
- **ML research:** Use `🧠 COTTON (ML/PyTorch)`
- **PDF processing:** Use `📄 AI-Papers (PDF Utils)`
- **Graph workflows:** Use `🕸️ Graph (LangGraph/LangFuse)`

### For Testing/Debugging
- **Quick tests:** Use `🚀 Root (LangChain RAG)` for minimal setup
- **Cross-environment tests:** Switch kernels within same notebook to compare

### For Production Notebooks
- **Deployment:** Ensure notebooks specify kernel explicitly
- **Documentation:** Include kernel requirements in notebook metadata

## 🔄 Workflow Examples

### Typical Development Session
```bash
# 1. Start in root directory
cd /mnt/d/llm/notebooks
echo $JUPYTER_DEFAULT_KERNEL  # → root-env

# 2. Move to specific project
cd langchain
echo $JUPYTER_DEFAULT_KERNEL  # → langchain-env

# 3. Start Jupyter
jupyter lab
# Creates new notebook with LangChain kernel pre-selected

# 4. Work on evaluation
cd ../deepeval/deepeval_claude_created
# Jupyter automatically suggests DeepEval kernel for new notebooks
```

### Multi-Environment Comparison
```bash
# 1. Open notebook in LangChain environment
cd langchain && jupyter lab notebook.ipynb

# 2. In Jupyter: Kernel → Change Kernel → 📊 DeepEval (Evaluation)
# Now same notebook runs with evaluation tools

# 3. Compare results between environments
```

### Research Workflow
```bash
# 1. Literature review with PDF processing
cd AI-Papers && jupyter lab
# Use 📄 AI-Papers kernel for PDF extraction

# 2. Implementation in ML environment  
cd ../Paper/1.3.3 && jupyter lab
# Switch to 🧠 COTTON kernel for PyTorch work

# 3. Evaluation
cd ../../deepeval/deepeval_claude_created
# Use 📊 DeepEval kernel for model assessment
```

## 💡 Best Practices

### 1. Consistent Environment Usage
- **Always check kernel** before running notebooks
- **Use directory-appropriate kernels** for optimal package availability
- **Document kernel requirements** in notebook markdown cells

### 2. Environment Isolation
- **Keep environments separate** - don't cross-install packages
- **Use shared environment** (graph-env) for related workflows
- **Regular cleanup** of unused kernels and environments

### 3. Collaboration
- **Include kernel info** in shared notebooks
- **Document environment setup** in project README
- **Standardize on kernel names** across team

### 4. Performance
- **Pre-warm environments** by running simple imports
- **Cache model downloads** using environment-specific cache dirs
- **Monitor resource usage** across different environments

This kernel management system ensures that each notebook automatically gets the right dependencies while maintaining clean environment separation! 🎯