#!/bin/bash

# Simple script to register Jupyter kernels for LangGraph and LangFuse modules
# This assumes virtual environments already exist or will be created manually

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Base directory
BASE_DIR="/home/admin88/notebook_clone"

# Function to register kernel if venv exists
register_kernel_if_exists() {
    local module_path="$1"
    local venv_name="$2"
    local kernel_display_name="$3"
    local kernel_name="$4"
    
    if [ -d "$module_path/venv_$venv_name" ]; then
        print_status "Registering kernel for $venv_name..."
        cd "$module_path"
        
        # Activate virtual environment and register kernel
        source "venv_$venv_name/bin/activate"
        
        # Install ipykernel if needed
        pip install ipykernel --quiet
        
        # Register the kernel
        python -m ipykernel install --user --name="$kernel_name" --display-name="$kernel_display_name"
        
        deactivate
        print_success "Registered: $kernel_display_name"
    else
        print_warning "Virtual environment not found: $module_path/venv_$venv_name"
        print_status "To create it manually:"
        echo "  cd $module_path"
        echo "  python3 -m venv venv_$venv_name"
        echo "  source venv_$venv_name/bin/activate"
        echo "  pip install -r requirements.txt"
        echo "  pip install ipykernel"
        echo "  python -m ipykernel install --user --name=$kernel_name --display-name=\"$kernel_display_name\""
    fi
}

# Function to create .envrc file for direnv
create_envrc_if_venv_exists() {
    local module_path="$1"
    local venv_name="$2"
    
    if [ -d "$module_path/venv_$venv_name" ]; then
        cd "$module_path"
        
        # Create .envrc file for direnv
        cat > .envrc << EOF
# Automatically activate virtual environment when entering directory
export VIRTUAL_ENV=\$(pwd)/venv_$venv_name
export PATH="\$VIRTUAL_ENV/bin:\$PATH"
unset PYTHON_HOME
EOF
        
        # Allow direnv to use this .envrc file
        if command -v direnv &> /dev/null; then
            direnv allow .
            print_success "Created .envrc file for $venv_name"
        else
            print_warning "direnv not found, .envrc created but not activated"
        fi
    fi
}

main() {
    print_status "Registering Jupyter kernels for existing virtual environments..."
    
    # Register LangGraph kernel if venv exists
    if [ -d "$BASE_DIR/langgraph" ]; then
        register_kernel_if_exists "$BASE_DIR/langgraph" "langgraph" "🕸️ LangGraph (Graph Workflows)" "langgraph_kernel"
        create_envrc_if_venv_exists "$BASE_DIR/langgraph" "langgraph"
    fi
    
    # Register LangFuse kernel if venv exists
    if [ -d "$BASE_DIR/langfuse" ]; then
        register_kernel_if_exists "$BASE_DIR/langfuse" "langfuse" "📊 LangFuse (Observability)" "langfuse_kernel"
        create_envrc_if_venv_exists "$BASE_DIR/langfuse" "langfuse"
    fi
    
    print_status "Checking available kernels..."
    
    # Try to list kernels if jupyter is available
    if command -v jupyter &> /dev/null; then
        jupyter kernelspec list
    else
        print_warning "Jupyter not found in PATH. Install jupyter in base environment to list kernels."
        echo "Kernel files should be in: ~/.local/share/jupyter/kernels/"
        ls -la ~/.local/share/jupyter/kernels/ 2>/dev/null || echo "Kernels directory not found yet."
    fi
    
    print_success "Kernel registration completed!"
    echo ""
    echo "Next steps:"
    echo "1. Navigate to langgraph/ or langfuse/ directories"
    echo "2. If using direnv, the virtual environment will activate automatically"
    echo "3. Start Jupyter: jupyter lab or jupyter notebook"
    echo "4. Select the appropriate kernel when creating notebooks"
}

main "$@"