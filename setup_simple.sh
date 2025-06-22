#!/bin/bash

# Simple setup script for direnv environments
set -e

echo "Setting up direnv environments..."

# Check if direnv is installed
if ! command -v direnv >/dev/null 2>&1; then
    echo "Error: direnv is not installed"
    echo "Install with: sudo apt install direnv"
    echo "Then add to ~/.bashrc: eval \"\$(direnv hook bash)\""
    exit 1
fi

echo "direnv is installed"

# Function to setup environment
setup_env() {
    local env_name=$1
    local env_path=$2
    local requirements_file=$3
    
    echo "Setting up $env_name environment..."
    
    if [[ -d "$env_path" ]]; then
        cd "$env_path"
        direnv allow
        
        if [[ -f "$requirements_file" ]]; then
            echo "Installing requirements..."
            pip install -r "$requirements_file"
        fi
        
        echo "$env_name environment setup complete"
        cd - >/dev/null
    else
        echo "Warning: Path not found: $env_path"
    fi
}

# Setup each environment
setup_env "Root" "." "requirements.txt"
setup_env "LangChain" "langchain" "requirements.txt"
setup_env "DeepEval" "deepeval/deepeval_claude_created" "requirements.txt"
setup_env "COTTON" "Paper/1.3.3" "requirements.txt"
setup_env "AI-Papers" "AI-Papers" "requirements.txt"

# Setup shared graph environment
echo "Setting up shared graph environment..."
cd ETL
direnv allow
pip install -r ../requirements_shared_graph.txt
cd ..

cd langfuse
direnv allow
cd ..

cd langgraph
direnv allow
cd ..

echo "Shared graph environment setup complete"

echo "All environments have been set up successfully!"
echo ""
echo "Next steps:"
echo "1. Run: ./register_kernels.sh"
echo "2. Run: python create_notebook_config.py"
echo ""
echo "Then navigate to any directory to activate its environment"