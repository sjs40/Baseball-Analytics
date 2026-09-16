from pathlib import Path


def data_path(layer: str) -> Path:
    path = Path.cwd() / "data" / layer
    path.mkdir(parents=True, exist_ok=True)
    return path
