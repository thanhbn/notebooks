#!/usr/bin/env python3
"""
Script to create .jupyter/notebook_config.py files for automatic kernel selection
This helps Jupyter automatically select the appropriate kernel based on notebook location
"""

import os
import json
from pathlib import Path

# Kernel mapping based on directory patterns
KERNEL_MAPPING = {
    # Exact directory matches (higher priority)
    "langchain": "langchain-env",
    "deepeval/deepeval_claude_created": "deepeval-env", 
    "Paper/1.3.3": "cotton-env",
    "AI-Papers": "ai-papers-env",
    "ETL": "graph-env",
    "langfuse": "graph-env",
    "langgraph": "graph-env",
    
    # Partial matches (lower priority)
    "deepeval": "deepeval-env",
    "Paper": "cotton-env",
}

def create_jupyter_config():
    """Create Jupyter configuration for automatic kernel selection"""
    
    config_content = '''# Jupyter Notebook Configuration
# Auto-generated configuration for kernel selection based on directory

import os
from pathlib import Path

def get_kernel_for_path(notebook_path):
    """Determine the appropriate kernel based on notebook path"""
    
    # Convert to Path object for easier manipulation
    path = Path(notebook_path).resolve()
    path_str = str(path)
    
    # Get relative path from project root for matching
    try:
        project_root = Path(__file__).parent.parent
        rel_path = path.relative_to(project_root)
        rel_path_str = str(rel_path)
    except:
        rel_path_str = path_str
    
    # Kernel mapping (most specific first)
    kernel_map = {
        "langchain": "langchain-env",
        "deepeval/deepeval_claude_created": "deepeval-env",
        "Paper/1.3.3": "cotton-env", 
        "AI-Papers": "ai-papers-env",
        "ETL": "graph-env",
        "langfuse": "graph-env", 
        "langgraph": "graph-env",
        "deepeval": "deepeval-env",
        "Paper": "cotton-env",
    }
    
    # Check for exact matches first
    for dir_pattern, kernel in kernel_map.items():
        if dir_pattern in rel_path_str or dir_pattern in path_str:
            return kernel
    
    # Default to root environment
    return "root-env"

# Configure default kernel selection
c = get_config()

# Set default kernel based on current directory
current_dir = os.getcwd()
default_kernel = get_kernel_for_path(current_dir)

c.MultiKernelManager.default_kernel_name = default_kernel

# Enable kernel selection logging
c.Application.log_level = 'INFO'

print(f"🧠 Jupyter configured to use kernel: {default_kernel}")
print(f"📁 Based on directory: {current_dir}")
'''
    
    return config_content

def setup_jupyter_configs():
    """Setup Jupyter configuration files in each directory"""
    
    base_dir = Path(".")
    
    # Directories that need specific configurations
    config_dirs = [
        ".",  # Root
        "langchain",
        "deepeval/deepeval_claude_created", 
        "Paper/1.3.3",
        "AI-Papers",
        "ETL",
        "langfuse",
        "langgraph"
    ]
    
    for config_dir in config_dirs:
        dir_path = base_dir / config_dir
        if dir_path.exists():
            # Create .jupyter directory
            jupyter_dir = dir_path / ".jupyter"
            jupyter_dir.mkdir(exist_ok=True)
            
            # Create notebook config
            config_file = jupyter_dir / "jupyter_notebook_config.py"
            
            # Determine kernel for this directory
            kernel_name = KERNEL_MAPPING.get(config_dir, "root-env")
            
            config_content = f'''# Jupyter Notebook Configuration for {config_dir}
# Auto-generated configuration

c = get_config()

# Set default kernel for this directory
c.MultiKernelManager.default_kernel_name = "{kernel_name}"

# Enable logging
c.Application.log_level = 'INFO'

print("🧠 Jupyter using kernel: {kernel_name}")
print("📁 Directory: {config_dir}")
'''
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            print(f"✅ Created config for {config_dir} → {kernel_name}")

def create_kernel_json_files():
    """Create kernel.json files with environment information"""
    
    kernels = {
        "root-env": {
            "display_name": "🚀 Root (LangChain RAG)",
            "language": "python",
            "metadata": {
                "debugger": True,
                "env_name": "root",
                "description": "Root environment for main LangChain RAG work"
            }
        },
        "langchain-env": {
            "display_name": "🦜 LangChain (Complete)", 
            "language": "python",
            "metadata": {
                "debugger": True,
                "env_name": "langchain",
                "description": "Comprehensive LangChain learning environment"
            }
        },
        "deepeval-env": {
            "display_name": "📊 DeepEval (Evaluation)",
            "language": "python", 
            "metadata": {
                "debugger": True,
                "env_name": "deepeval",
                "description": "AI model evaluation and benchmarking"
            }
        },
        "cotton-env": {
            "display_name": "🧠 COTTON (ML/PyTorch)",
            "language": "python",
            "metadata": {
                "debugger": True, 
                "env_name": "cotton",
                "description": "Machine learning research environment"
            }
        },
        "ai-papers-env": {
            "display_name": "📄 AI-Papers (PDF Utils)",
            "language": "python",
            "metadata": {
                "debugger": True,
                "env_name": "ai_papers", 
                "description": "PDF processing and text extraction"
            }
        },
        "graph-env": {
            "display_name": "🕸️ Graph (LangGraph/LangFuse)",
            "language": "python",
            "metadata": {
                "debugger": True,
                "env_name": "graph",
                "description": "Shared graph-based applications environment"
            }
        }
    }
    
    # Create kernel specs directory
    kernel_specs_dir = Path.home() / ".local/share/jupyter/kernels"
    kernel_specs_dir.mkdir(parents=True, exist_ok=True)
    
    for kernel_name, kernel_spec in kernels.items():
        kernel_dir = kernel_specs_dir / kernel_name
        kernel_dir.mkdir(exist_ok=True)
        
        # Create kernel.json
        kernel_json = kernel_dir / "kernel.json"
        with open(kernel_json, 'w') as f:
            json.dump(kernel_spec, f, indent=2)
        
        print(f"📋 Created kernel spec: {kernel_name}")

if __name__ == "__main__":
    print("🔧 Setting up Jupyter configurations for automatic kernel selection")
    print("="*70)
    
    print("\n📁 Creating directory-specific configurations...")
    setup_jupyter_configs()
    
    print("\n📋 Creating kernel specifications...")
    create_kernel_json_files()
    
    print("\n🎉 Jupyter configuration complete!")
    print("\n💡 Usage:")
    print("   1. Start Jupyter from any directory: jupyter lab or jupyter notebook")
    print("   2. New notebooks will automatically use the appropriate kernel")
    print("   3. Existing notebooks can be changed via Kernel → Change Kernel menu")
    print("\n🔄 To reconfigure, run this script again or use:")
    print("   python create_notebook_config.py")