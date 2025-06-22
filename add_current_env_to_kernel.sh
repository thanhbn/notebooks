#!/bin/bash

# Script to detect current virtual environment and add it to IPython kernel
# Works with direnv to automatically register the active environment

set -e

echo "🔍 Detecting current virtual environment..."
echo "==========================================="

# Function to extract kernel name from environment
get_kernel_name() {
    local env_path=$1
    local kernel_name=""
    
    # Check common virtual environment patterns
    if [[ "$env_path" == *"/venv_root"* ]]; then
        kernel_name="root-env"
    elif [[ "$env_path" == *"/venv_langchain"* ]]; then
        kernel_name="langchain-env"
    elif [[ "$env_path" == *"/venv_deepeval"* ]]; then
        kernel_name="deepeval-env"
    elif [[ "$env_path" == *"/venv_cotton"* ]]; then
        kernel_name="cotton-env"
    elif [[ "$env_path" == *"/venv_ai_papers"* ]]; then
        kernel_name="ai-papers-env"
    elif [[ "$env_path" == *"/venv_shared_graph"* ]]; then
        kernel_name="graph-env"
    else
        # Use directory name as fallback
        kernel_name=$(basename "$env_path")
    fi
    
    echo "$kernel_name"
}

# Function to get display name based on kernel name
get_display_name() {
    local kernel_name=$1
    case "$kernel_name" in
        "root-env")
            echo "🚀 Root (LangChain RAG)"
            ;;
        "langchain-env")
            echo "🦜 LangChain (Complete)"
            ;;
        "deepeval-env")
            echo "📊 DeepEval (Evaluation)"
            ;;
        "cotton-env")
            echo "🧠 COTTON (ML/PyTorch)"
            ;;
        "ai-papers-env")
            echo "📄 AI-Papers (PDF Utils)"
            ;;
        "graph-env")
            echo "🕸️ Graph (LangGraph/LangFuse)"
            ;;
        *)
            echo "🐍 $kernel_name"
            ;;
    esac
}

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ No virtual environment detected!"
    echo ""
    echo "💡 Please activate a virtual environment first:"
    echo "   • Using direnv: cd to a directory with .envrc file"
    echo "   • Manual activation: source <venv_path>/bin/activate"
    echo ""
    echo "📁 Available environments in this project:"
    echo "   • venv_root - Root environment"
    echo "   • langchain/venv_langchain - LangChain environment"
    echo "   • deepeval/deepeval_claude_created/venv_deepeval - DeepEval environment"
    echo "   • Paper/1.3.3/venv_cotton - COTTON environment"
    echo "   • AI-Papers/venv_ai_papers - AI Papers environment"
    echo "   • venv_shared_graph - Shared graph environment"
    exit 1
fi

# Get environment details
ENV_PATH="$VIRTUAL_ENV"
ENV_NAME="${ENV_NAME:-$(basename "$VIRTUAL_ENV")}"
KERNEL_NAME=$(get_kernel_name "$ENV_PATH")
DISPLAY_NAME=$(get_display_name "$KERNEL_NAME")

echo "✅ Virtual environment detected:"
echo "   Path: $ENV_PATH"
echo "   Name: $ENV_NAME"
echo "   Kernel name: $KERNEL_NAME"
echo "   Display name: $DISPLAY_NAME"
echo ""

# Check if ipykernel is installed
echo "📦 Checking for ipykernel..."
if ! python -c "import ipykernel" 2>/dev/null; then
    echo "   Installing ipykernel..."
    pip install ipykernel -q
    echo "   ✅ ipykernel installed"
else
    echo "   ✅ ipykernel already installed"
fi

# Check if kernel already exists
echo ""
echo "🔍 Checking existing kernels..."
if jupyter kernelspec list 2>/dev/null | grep -q "$KERNEL_NAME"; then
    echo "   ⚠️  Kernel '$KERNEL_NAME' already exists"
    echo ""
    read -p "   Do you want to update it? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Skipping kernel registration"
        exit 0
    fi
    echo "   Removing existing kernel..."
    jupyter kernelspec remove "$KERNEL_NAME" -f
fi

# Register the kernel
echo ""
echo "📋 Registering kernel..."
python -m ipykernel install --user \
    --name="$KERNEL_NAME" \
    --display-name="$DISPLAY_NAME"

echo ""
echo "✅ Kernel registered successfully!"
echo ""

# Show current kernel list
echo "📋 Current kernels:"
jupyter kernelspec list | grep -E "(Available|$KERNEL_NAME|python3)"
echo ""

# Show usage instructions
echo "💡 Usage:"
echo "   1. Start Jupyter: jupyter notebook or jupyter lab"
echo "   2. Select kernel: $DISPLAY_NAME"
echo "   3. Or use from command line:"
echo "      jupyter console --kernel=$KERNEL_NAME"
echo ""

# If JUPYTER_DEFAULT_KERNEL is set, show it
if [[ -n "$JUPYTER_DEFAULT_KERNEL" ]]; then
    echo "🎯 Default kernel for this directory: $JUPYTER_DEFAULT_KERNEL"
    if [[ "$JUPYTER_DEFAULT_KERNEL" != "$KERNEL_NAME" ]]; then
        echo "   ⚠️  Note: Current env kernel ($KERNEL_NAME) differs from default"
    fi
fi

echo ""
echo "🎉 Done!"