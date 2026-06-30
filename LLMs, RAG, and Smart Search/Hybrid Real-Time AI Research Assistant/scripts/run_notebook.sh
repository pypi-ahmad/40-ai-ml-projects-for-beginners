#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m jupyter nbconvert --to notebook --execute notebooks/project_18_hybrid_real_time_ai_research_assistant.ipynb --output project_18_hybrid_real_time_ai_research_assistant.executed.ipynb
