"""Resolve repository resources independently of the working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
MODEL_ROOT = PROJECT_ROOT / "models"
SAMPLE_ROOT = PROJECT_ROOT / "samples" / "judges"
