# Manual Setup Guide

If the automated scripts have issues, follow these manual steps:

## 1. Install direnv

```bash
# Ubuntu/Debian
sudo apt install direnv

# Add to ~/.bashrc or ~/.zshrc
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
# or for zsh
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Restart shell or source config
source ~/.bashrc
```

## 2. Create Virtual Environments

```bash
cd /mnt/d/llm/notebooks

# Root environment
python3 -m venv venv_root
source venv_root/bin/activate
pip install -r requirements.txt
deactivate

# LangChain environment
cd langchain
python3 -m venv venv_langchain
source venv_langchain/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# DeepEval environment
cd deepeval/deepeval_claude_created
python3 -m venv venv_deepeval
source venv_deepeval/bin/activate
pip install -r requirements.txt
deactivate
cd ../..

# COTTON environment
cd Paper/1.3.3
python3 -m venv venv_cotton
source venv_cotton/bin/activate
pip install -r requirements.txt
deactivate
cd ../..

# AI-Papers environment
cd AI-Papers
python3 -m venv venv_ai_papers
source venv_ai_papers/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Shared graph environment
python3 -m venv venv_shared_graph
source venv_shared_graph/bin/activate
pip install -r requirements_shared_graph.txt
deactivate
```

## 3. Allow direnv for each directory

```bash
cd /mnt/d/llm/notebooks
direnv allow

cd langchain
direnv allow
cd ..

cd deepeval/deepeval_claude_created
direnv allow
cd ../..

cd Paper/1.3.3
direnv allow
cd ../..

cd AI-Papers
direnv allow
cd ..

cd ETL
direnv allow
cd ..

cd langfuse
direnv allow
cd ..

cd langgraph
direnv allow
cd ..
```

## 4. Register Jupyter Kernels

```bash
cd /mnt/d/llm/notebooks

# Register each kernel manually
source venv_root/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="root-env" --display-name="🚀 Root (LangChain RAG)"
deactivate

source langchain/venv_langchain/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="langchain-env" --display-name="🦜 LangChain (Complete)"
deactivate

source deepeval/deepeval_claude_created/venv_deepeval/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="deepeval-env" --display-name="📊 DeepEval (Evaluation)"
deactivate

source Paper/1.3.3/venv_cotton/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="cotton-env" --display-name="🧠 COTTON (ML/PyTorch)"
deactivate

source AI-Papers/venv_ai_papers/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="ai-papers-env" --display-name="📄 AI-Papers (PDF Utils)"
deactivate

source venv_shared_graph/bin/activate
pip install ipykernel
python -m ipykernel install --user --name="graph-env" --display-name="🕸️ Graph (LangGraph/LangFuse)"
deactivate
```

## 5. Verify Setup

```bash
# Check kernels
jupyter kernelspec list

# Test environment activation
cd langchain
echo $ENV_NAME  # Should show "langchain"
echo $JUPYTER_DEFAULT_KERNEL  # Should show "langchain-env"

cd ../deepeval/deepeval_claude_created
echo $ENV_NAME  # Should show "deepeval"
```

## 6. Test Jupyter

```bash
cd langchain
jupyter lab
# Create new notebook - should suggest LangChain kernel

cd ../deepeval/deepeval_claude_created
jupyter lab
# Create new notebook - should suggest DeepEval kernel
```

## Troubleshooting

### Environment not activating
```bash
# Check direnv status
direnv status

# Reload direnv
direnv reload

# Check .envrc exists
ls -la .envrc
```

### Kernel not found
```bash
# List kernels
jupyter kernelspec list

# Remove and re-register
jupyter kernelspec remove root-env
# Then follow step 4 again
```

### Package not found
```bash
# Check which environment is active
which python
echo $VIRTUAL_ENV

# Activate correct environment and install
source venv_name/bin/activate
pip install package_name
```