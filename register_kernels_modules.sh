#!/bin/bash

# Script to register Jupyter kernels for LangGraph and LangFuse modules
# This script sets up isolated environments and registers them as Jupyter kernels

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/admin88/notebook_clone"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if virtual environment exists
check_venv() {
    local venv_path="$1"
    if [ -d "$venv_path" ]; then
        return 0
    else
        return 1
    fi
}

# Function to create virtual environment if it doesn't exist
create_venv() {
    local module_path="$1"
    local venv_name="$2"
    
    cd "$module_path"
    
    if ! check_venv "venv_$venv_name"; then
        print_status "Creating virtual environment for $venv_name..."
        python3 -m venv "venv_$venv_name"
        print_success "Virtual environment created for $venv_name"
    else
        print_warning "Virtual environment already exists for $venv_name"
    fi
}

# Function to install requirements and register kernel
setup_kernel() {
    local module_path="$1"
    local venv_name="$2"
    local kernel_display_name="$3"
    local kernel_name="$4"
    
    print_status "Setting up kernel for $venv_name..."
    
    cd "$module_path"
    
    # Activate virtual environment
    source "venv_$venv_name/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        print_status "Installing requirements for $venv_name..."
        pip install -r requirements.txt
        print_success "Requirements installed for $venv_name"
    else
        print_warning "No requirements.txt found for $venv_name"
    fi
    
    # Install ipykernel if not already installed
    pip install ipykernel
    
    # Register the kernel
    print_status "Registering Jupyter kernel: $kernel_display_name"
    python -m ipykernel install --user --name="$kernel_name" --display-name="$kernel_display_name"
    
    # Deactivate virtual environment
    deactivate
    
    print_success "Kernel registered: $kernel_display_name"
}

# Function to create .envrc file for direnv
create_envrc() {
    local module_path="$1"
    local venv_name="$2"
    
    cd "$module_path"
    
    # Create .envrc file for direnv
    cat > .envrc << EOF
# Automatically activate virtual environment when entering directory
export VIRTUAL_ENV=\$(pwd)/venv_$venv_name
export PATH="\$VIRTUAL_ENV/bin:\$PATH"
unset PYTHON_HOME
EOF
    
    # Allow direnv to use this .envrc file
    direnv allow .
    
    print_success "Created .envrc file for $venv_name"
}

# Main execution
main() {
    print_status "Starting Jupyter kernel registration for LangGraph and LangFuse modules..."
    
    # Check if we're in the right directory
    if [ ! -d "$BASE_DIR" ]; then
        print_error "Base directory $BASE_DIR not found!"
        exit 1
    fi
    
    # Setup LangGraph module
    if [ -d "$BASE_DIR/langgraph" ]; then
        print_status "Processing LangGraph module..."
        create_venv "$BASE_DIR/langgraph" "langgraph"
        setup_kernel "$BASE_DIR/langgraph" "langgraph" "🕸️ LangGraph (Graph Workflows)" "langgraph_kernel"
        create_envrc "$BASE_DIR/langgraph" "langgraph"
        print_success "LangGraph module setup completed"
    else
        print_warning "LangGraph directory not found, skipping..."
    fi
    
    # Setup LangFuse module
    if [ -d "$BASE_DIR/langfuse" ]; then
        print_status "Processing LangFuse module..."
        create_venv "$BASE_DIR/langfuse" "langfuse"
        setup_kernel "$BASE_DIR/langfuse" "langfuse" "📊 LangFuse (Observability)" "langfuse_kernel"
        create_envrc "$BASE_DIR/langfuse" "langfuse"
        print_success "LangFuse module setup completed"
    else
        print_warning "LangFuse directory not found, skipping..."
    fi
    
    print_success "All kernels registered successfully!"
    print_status "Available kernels:"
    jupyter kernelspec list
    
    print_status "Usage instructions:"
    echo "1. Navigate to langgraph/ or langfuse/ directories"
    echo "2. The virtual environment will activate automatically (via direnv)"
    echo "3. Start Jupyter: jupyter lab or jupyter notebook"
    echo "4. Select the appropriate kernel when creating notebooks"
    
    print_success "Setup completed! You can now use the specialized kernels."
}

# Check if direnv is installed
if ! command -v direnv &> /dev/null; then
    print_error "direnv is not installed. Please install it first:"
    echo "  Ubuntu/Debian: sudo apt install direnv"
    echo "  macOS: brew install direnv"
    echo "  Then add 'eval \"\$(direnv hook bash)\"' to your ~/.bashrc"
    exit 1
fi

# Run main function
main "$@"