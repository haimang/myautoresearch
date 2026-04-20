import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"

if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))
