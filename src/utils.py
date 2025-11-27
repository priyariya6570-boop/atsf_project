from pathlib import Path

def get_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / filename
