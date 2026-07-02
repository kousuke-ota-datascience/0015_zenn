from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
API_ROOT = APP_ROOT.parent
PROJECT_ROOT = API_ROOT.parent

CATALOG_PATH = APP_ROOT / "datasets.yaml"
DATA_ROOT = PROJECT_ROOT
