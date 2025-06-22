#!/bin/bash

# Setup script for direnv environments in LangChain notebooks repository
# This script sets up all virtual environments and installs requirements

set -e  # Exit on any error

echo "🚀 Setting up direnv environments for LangChain notebooks repository"
echo "========================================================================"

# Check if direnv is installed
if ! command -v direnv &> /dev/null; then
    echo "❌ direnv is not installed. Please install direnv first:"
    echo "   - Ubuntu/Debian: sudo apt install direnv"
    echo "   - macOS: brew install direnv"
    echo "   - Then add 'eval \"\$(direnv hook bash)\"' to your ~/.bashrc or ~/.zshrc"
    exit 1
fi

echo "✅ direnv is installed"

# Function to setup environment
setup_env() {
    local env_name=$1
    local env_path=$2
    local requirements_file=$3
    
    echo ""
    echo "📦 Setting up $env_name environment..."
    echo "   Path: $env_path"
    echo "   Requirements: $requirements_file"
    
    cd "$env_path"
    
    # Allow direnv
    direnv allow
    
    # Install requirements if file exists
    if [[ -f "$requirements_file" ]]; then
        echo "   Installing requirements..."
        pip install -r "$requirements_file"
    else
        echo "   ⚠️  Requirements file not found: $requirements_file"
    fi
    
    echo "   ✅ $env_name environment setup complete"
    cd - > /dev/null
}

# Setup root environment
setup_env "Root" "." "requirements.txt"

# Setup LangChain environment
setup_env "LangChain" "langchain" "requirements.txt"

# Setup DeepEval environment
setup_env "DeepEval" "deepeval/deepeval_claude_created" "requirements.txt"

# Setup COTTON environment
setup_env "COTTON" "Paper/1.3.3" "requirements.txt"

# Setup AI-Papers environment
setup_env "AI-Papers" "AI-Papers" "requirements.txt"

# Setup shared graph environment (ETL, LangFuse, LangGraph)
echo ""
echo "📊 Setting up shared graph environment for ETL/LangFuse/LangGraph..."

# Setup ETL
cd ETL
direnv allow
pip install -r ../requirements_shared_graph.txt
cd ..

# Setup LangFuse
cd langfuse
direnv allow
cd ..

# Setup LangGraph
cd langgraph
direnv allow
cd ..

echo "   ✅ Shared graph environment setup complete"

echo ""
echo "🎉 All environments have been set up successfully!"

# Register Jupyter kernels
echo ""
echo "🧠 Registering Jupyter kernels..."
./register_kernels.sh

# Create Jupyter configurations
echo ""
echo "🔧 Creating Jupyter configurations..."
python create_notebook_config.py

echo ""
echo "📋 Summary of environments:"
echo "   1. Root (/) - Basic LangChain RAG setup → 🚀 Root (LangChain RAG)"
echo "   2. LangChain (/langchain) - Comprehensive LangChain learning → 🦜 LangChain (Complete)"
echo "   3. DeepEval (/deepeval/deepeval_claude_created) - Model evaluation → 📊 DeepEval (Evaluation)"
echo "   4. COTTON (/Paper/1.3.3) - ML/PyTorch for code generation → 🧠 COTTON (ML/PyTorch)"
echo "   5. AI-Papers (/AI-Papers) - PDF processing utilities → 📄 AI-Papers (PDF Utils)"
echo "   6. Shared (/ETL, /langfuse, /langgraph) - Graph-based applications → 🕸️ Graph (LangGraph/LangFuse)"
echo ""
echo "💡 Usage:"
echo "   - Navigate to any directory to automatically activate its environment"
echo "   - Use 'direnv allow' to permit new .envrc files"
echo "   - Use 'direnv deny' to disable direnv for a directory"
echo "   - Jupyter notebooks will automatically suggest the correct kernel for each directory"
echo ""
echo "🧠 Jupyter Integration:"
echo "   - Each directory has a registered Jupyter kernel"
echo "   - New notebooks will show kernel suggestions based on location"
echo "   - Use Kernel → Change Kernel menu to switch if needed"
echo ""
echo "🔧 To customize environments:"
echo "   - Edit the .envrc files in each directory"
echo "   - Modify requirements.txt files as needed"
echo "   - Run 'direnv reload' after making changes"
echo "   - Re-run './register_kernels.sh' after adding new dependencies to kernels"