#!/usr/bin/env bash
# ZIM RAG Setup Script
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${ZIM_RAG_LLM_MODEL:-ai/smollm3:Q4_K_M}"

echo "Setting up ZIM to Vector Database RAG System..."
echo "================================================"

if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3.10+ is required but python3 was not found."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo "Error: Python 3.10+ is required, found $(python3 --version)."
    exit 1
fi
python3 --version

if [ ! -d zim_rag_env ]; then
    echo "Creating virtual environment..."
    python3 -m venv zim_rag_env
fi
# shellcheck disable=SC1091
source zim_rag_env/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p vector_db zim_library

# Local LLM via Docker Model Runner (optional, needed only for rag-query/chat).
if command -v docker &> /dev/null && docker model version &> /dev/null; then
    echo "Docker Model Runner found. Pulling ${MODEL}..."
    docker model pull "${MODEL}" || echo "Could not pull ${MODEL}; run 'docker model pull ${MODEL}' later."
else
    echo "Docker Model Runner not found. For local answers install Docker Desktop and"
    echo "enable Model Runner, or configure Ollama, OpenAI or Anthropic in config.json."
fi

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source zim_rag_env/bin/activate"
echo "2. Add ZIM files to ./zim_library/ (download from https://library.kiwix.org/)"
echo "3. Build the vector database: python zim_rag.py build"
echo "4. Search: python zim_rag.py query \"your question here\""
echo "5. Ask the LLM: python zim_rag.py rag-query \"your question here\""
echo ""
echo "For more options, see: python zim_rag.py --help"
