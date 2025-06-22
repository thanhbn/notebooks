#!/bin/bash

# Script to register Jupyter kernels for each virtual environment
# This ensures notebooks can automatically use the correct Python environment

set -e

echo "🧠 Registering Jupyter kernels for all virtual environments"
echo "============================================================"

# Function to register a kernel
register_kernel() {
    local kernel_name=$1
    local env_path=$2
    local display_name=$3
    local description=$4
    
    echo ""
    echo "📋 Registering kernel: $kernel_name"
    echo "   Environment: $env_path"
    echo "   Display name: $display_name"
    
    if [[ -d "$env_path" ]]; then
        # Activate environment and register kernel
        source "$env_path/bin/activate"
        
        # Install ipykernel if not present
        pip install ipykernel -q
        
        # Register the kernel with custom display name
        python -m ipykernel install --user --name="$kernel_name" --display-name="$display_name"
        
        echo "   ✅ Kernel '$kernel_name' registered successfully"
        deactivate
    else
        echo "   ⚠️  Environment not found: $env_path"
        echo "      Run ./setup_direnv.sh first to create environments"
    fi
}

# Register kernels for each environment
register_kernel "root-env" "venv_root" "🚀 Root (LangChain RAG)" "Root environment for main LangChain RAG work"

register_kernel "langchain-env" "langchain/venv_langchain" "🦜 LangChain (Complete)" "Comprehensive LangChain learning environment"

register_kernel "deepeval-env" "deepeval/deepeval_claude_created/venv_deepeval" "📊 DeepEval (Evaluation)" "AI model evaluation and benchmarking"

register_kernel "cotton-env" "Paper/1.3.3/venv_cotton" "🧠 COTTON (ML/PyTorch)" "Machine learning research environment"

register_kernel "ai-papers-env" "AI-Papers/venv_ai_papers" "📄 AI-Papers (PDF Utils)" "PDF processing and text extraction"

register_kernel "graph-env" "venv_shared_graph" "🕸️ Graph (LangGraph/LangFuse)" "Shared graph-based applications environment"

echo ""
echo "🎉 All kernels registered successfully!"
echo ""
echo "📋 Available kernels:"
jupyter kernelspec list
echo ""
echo "💡 Usage in Jupyter:"
echo "   1. Open Jupyter Lab/Notebook in any directory"
echo "   2. When creating/opening notebooks, select the appropriate kernel:"
echo "      • 🚀 Root (LangChain RAG) - for main.py and root notebooks"
echo "      • 🦜 LangChain (Complete) - for /langchain/ notebooks"
echo "      • 📊 DeepEval (Evaluation) - for /deepeval/ notebooks"
echo "      • 🧠 COTTON (ML/PyTorch) - for /Paper/1.3.3/ notebooks"
echo "      • 📄 AI-Papers (PDF Utils) - for /AI-Papers/ notebooks"
echo "      • 🕸️ Graph (LangGraph/LangFuse) - for /ETL/, /langfuse/, /langgraph/ notebooks"
echo ""
echo "🔄 To change kernel in existing notebook:"
echo "   • Jupyter Lab: Kernel menu → Change Kernel"
echo "   • Jupyter Notebook: Kernel menu → Change kernel"
echo ""
echo "🧹 To remove kernels later:"
echo "   jupyter kernelspec remove <kernel-name>"