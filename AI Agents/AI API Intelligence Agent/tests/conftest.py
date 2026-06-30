import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("AGENT_DISABLE_CHROMA", "1")
