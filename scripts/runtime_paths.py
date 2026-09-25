import os
from pathlib import Path

WORKSPACE_ROOT = Path(
    os.environ.get("MODEL_TRAINER_WORKSPACE", Path(__file__).resolve().parent.parent)
)
