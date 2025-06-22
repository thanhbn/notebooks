#!/bin/bash

# Setup script for LangChain learning environment
# Run this from the project root: /mnt/d/llm/notebooks/

echo "🚀 Starting LangChain environment setup..."

# Check if virtual environment exists
if [ -d "langchain_env" ]; then
    echo "⚠️  Virtual environment already exists. Activating..."
else
    echo "📦 Creating virtual environment..."
    python3 -m venv langchain_env
fi

# Activate virtual environment
source langchain_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Jupyter
echo "📦 Installing Jupyter..."
pip install jupyter ipykernel

# Add Jupyter kernel
echo "🔧 Adding Jupyter kernel..."
python -m ipykernel install --user --name=langchain_env --display-name='Python (langchain_env)'

# Install requirements if file exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
else
    echo "📦 Installing basic packages..."
    pip install langchain langchain-anthropic python-dotenv
fi

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source langchain_env/bin/activate"
echo ""
echo "To start Jupyter with the correct kernel:"
echo "jupyter notebook"
