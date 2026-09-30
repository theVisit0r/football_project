from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

FBREF_DATA_DIR = DATA_DIR / "scraped"
SOCCERDATA_DIR = DATA_DIR / "soccerdata"

RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"

DB_PATH = PROJECT_ROOT / "football_project.db"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"