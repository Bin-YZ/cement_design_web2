from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
ASSETS_DIR = PROJECT_DIR / "assets"
MODELS_DIR = PROJECT_DIR / "models"
DOCS_DIR = PROJECT_DIR / "docs"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


def get_logo_path() -> Path:
    return ASSETS_DIR / "psi_logo.png"


def get_model_path() -> Path:
    return MODELS_DIR / "my_model16082025.h5"
