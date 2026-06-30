#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/documents/bootstrap

cat > data/documents/bootstrap/README.md << 'DOC'
# Corpus Bootstrap Instructions

Add high-quality sources here, such as:
- Python documentation PDFs/HTML exports
- LangChain/LangGraph docs
- Scikit-learn user guide
- FastAPI docs
- CUDA docs
- Linux docs/man pages
- AI papers/reports

Use legal/publicly distributable copies only.
DOC

echo "Created corpus bootstrap instructions at data/documents/bootstrap/README.md"
