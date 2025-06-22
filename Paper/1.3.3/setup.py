#!/usr/bin/env python3
"""
COTTON Setup Script
Automated setup for the COTTON implementation
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    COTTON Implementation                     ║
    ║        Chain-of-Thought in Neural Code Generation            ║
    ║                                                              ║
    ║   Paper: Yang et al., IEEE Trans. Software Engineering      ║
    ║   Implementation: LangChain + LangGraph + DeepEval          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU detected: {gpu_name} ({memory:.1f}GB)")
            print(f"✅ CUDA version: {torch.version.cuda}")
            return True, gpu_name, memory
        else:
            print("⚠️  No GPU detected. CPU-only mode will be used.")
            return False, None, 0
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be performed after installation.")
        return False, None, 0

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.10.0",
        "peft>=0.4.0",
        "langchain>=0.0.350",
        "numpy>=1.21.0",
        "pandas>=1.5.0"
    ]
    
    # Optional but recommended
    optional_packages = [
        "langgraph>=0.0.20",
        "deepeval>=0.20.0",
        "nltk>=3.8",
        "rouge-score>=0.1.2",
        "matplotlib>=3.5.0",
        "jupyter>=1.0.0"
    ]
    
    print("Installing core dependencies...")
    for package in core_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\nInstalling optional dependencies...")
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional package {package} failed to install")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    directories = [
        "cotton_data",
        "cotton_model", 
        "logs",
        "results",
        "checkpoints"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created {dir_name}/")

def download_sample_data():
    """Download sample datasets"""
    print("\n🔄 Setting up sample data...")
    try:
        from datasets import load_dataset
        
        # Download HumanEval dataset
        print("Downloading HumanEval dataset...")
        dataset = load_dataset("openai/HumanEval", split="test")
        
        # Save sample data
        sample_data = []
        for i, item in enumerate(dataset):
            if i >= 20:  # Limit for demo
                break
            sample_data.append({
                'task_id': item['task_id'],
                'prompt': item['prompt'],
                'canonical_solution': item['canonical_solution'],
                'test': item['test']
            })
        
        with open("cotton_data/sample_humaneval.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Downloaded {len(sample_data)} sample problems")
        
    except Exception as e:
        print(f"⚠️  Sample data download failed: {e}")
        print("You can run the setup manually later.")

def create_config_file():
    """Create default configuration file"""
    print("\n⚙️  Creating configuration file...")
    try:
        from config import COTTONConfig, EnvironmentConfigs
        
        # Detect best configuration based on hardware
        has_gpu, gpu_name, gpu_memory = check_gpu()
        
        if has_gpu and gpu_memory >= 20:  # RTX 3090/4090 or better
            config = EnvironmentConfigs.rtx_3090_config()
            print(f"✅ Using RTX 3090/4090 optimized configuration")
        elif has_gpu:
            config = EnvironmentConfigs.cpu_only_config()
            config.device = "cuda:0"
            print(f"✅ Using GPU configuration with reduced settings")
        else:
            config = EnvironmentConfigs.cpu_only_config()
            print(f"✅ Using CPU-only configuration")
        
        # Save configuration
        config.save("cotton_config.json")
        print("✅ Configuration saved to cotton_config.json")
        
    except Exception as e:
        print(f"⚠️  Configuration creation failed: {e}")

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    test_script = """
import sys
sys.path.append('.')

# Test core imports
try:
    import torch
    import transformers
    import datasets
    import peft
    import langchain
    print("✅ Core dependencies imported successfully")
except ImportError as e:
    print(f"❌ Core import failed: {e}")
    sys.exit(1)

# Test optional imports
optional_imports = []
try:
    import langgraph
    optional_imports.append("langgraph")
except ImportError:
    pass

try:
    import deepeval
    optional_imports.append("deepeval")
except ImportError:
    pass

try:
    from cotton_implementation import COTTONConfig, DataCleaner, COTTONEvaluator
    print("✅ COTTON implementation imported successfully")
except ImportError as e:
    print(f"❌ COTTON import failed: {e}")

print(f"✅ Optional packages available: {', '.join(optional_imports) if optional_imports else 'None'}")
print("🎉 Installation test completed!")
"""
    
    try:
        exec(test_script)
    except Exception as e:
        print(f"❌ Installation test failed: {e}")

def create_quick_start_script():
    """Create a quick start script"""
    print("\n📝 Creating quick start script...")
    
    quick_start = '''#!/usr/bin/env python3
"""
Quick Start Script for COTTON Implementation
"""

import sys
sys.path.append('.')

from cotton_implementation import main_cotton_pipeline, COTTONConfig
from config import COTTONConfigVariants

def main():
    print("🚀 COTTON Quick Start")
    print("=" * 50)
    
    # Choose configuration
    print("Select configuration:")
    print("1. Demo (20 samples, 2 epochs)")
    print("2. Research (1000 samples, 10 epochs)")
    print("3. Production (9000 samples, 20 epochs)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        config = COTTONConfigVariants.demo_config()
        print("✅ Using demo configuration")
    elif choice == "2":
        config = COTTONConfigVariants.research_config()
        print("✅ Using research configuration")
    elif choice == "3":
        config = COTTONConfigVariants.production_config()
        print("✅ Using production configuration")
    else:
        print("Invalid choice. Using demo configuration.")
        config = COTTONConfigVariants.demo_config()
    
    # Run pipeline
    print("\\n🔄 Running COTTON pipeline...")
    results = main_cotton_pipeline()
    
    print("\\n📊 Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print("\\n🎉 Quick start completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w") as f:
        f.write(quick_start)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("quick_start.py", 0o755)
    
    print("✅ Quick start script created: quick_start.py")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 COTTON Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. Run demo: python quick_start.py")
    print("2. Open Jupyter notebook: jupyter notebook cotton_notebook.ipynb")
    print("3. Review configuration: cat cotton_config.json")
    print("4. Read documentation: cat README.md")
    
    print("\n🔧 Files Created:")
    files = [
        "cotton_implementation.py - Main implementation",
        "config.py - Configuration management", 
        "cotton_config.json - Current configuration",
        "quick_start.py - Quick start script",
        "cotton_notebook.ipynb - Jupyter notebook",
        "README.md - Documentation",
        "requirements.txt - Dependencies"
    ]
    
    for file in files:
        if os.path.exists(file.split(" - ")[0]):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    print("\n💡 Tips:")
    print("- For full training, ensure you have sufficient GPU memory (24GB+)")
    print("- Modify cotton_config.json for custom configurations")
    print("- Check logs/ directory for detailed execution logs")
    print("- Use CPU-only mode for testing without GPU")
    
    print("\n📚 Resources:")
    print("- Paper: IEEE Transactions on Software Engineering 2024")
    print("- LangChain docs: https://langchain.readthedocs.io/")
    print("- DeepEval docs: https://docs.confident-ai.com/")
    print("- Transformers docs: https://huggingface.co/docs/transformers/")

def main():
    """Main setup function"""
    print_banner()
    
    # Pre-installation checks
    check_python_version()
    check_gpu()
    
    # Setup process
    try:
        install_dependencies()
        setup_directories()
        download_sample_data()
        create_config_file()
        test_installation()
        create_quick_start_script()
        
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and run setup again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
