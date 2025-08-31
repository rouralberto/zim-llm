#!/bin/bash

# ZIM RAG Setup Script

echo "Setting up ZIM to Vector Database RAG System..."
echo "================================================"

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv zim_rag_env
source zim_rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install ZIM library support
echo "Installing ZIM library support..."
pip install libzim 2>/dev/null || echo "libzim not available (requires system libraries)"

# Check if Ollama is available
if command -v ollama &> /dev/null; then
    echo "Ollama found. Pulling llama2 model..."
    ollama pull llama2
else
    echo "Ollama not found. Install from https://ollama.ai for local LLM support."
fi

# Create necessary directories
mkdir -p vector_db
mkdir -p zim_library

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source zim_rag_env/bin/activate"
echo "2. Add ZIM files to ./zim_library/ (download from https://library.kiwix.org/)"
echo "3. Build the vector database: python zim_rag.py build"
echo "4. Query the knowledge base: python zim_rag.py query \"your question here\""
echo ""
echo "For more options, see: python zim_rag.py --help"
