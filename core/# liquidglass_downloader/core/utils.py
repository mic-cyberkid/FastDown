# fastdown/core/utils.py
from pathlib import Path
import sys
import multiprocessing


def safe_output_path(user_path: str, default_dir: Path | None = None) -> Path:
    path = Path(user_path)
    if not path.is_absolute():
        path = (default_dir or Path.home() / "Downloads") / path
    resolved = path.resolve()

    safe_roots = [Path.home()] + ([Path("/tmp")] if sys.platform != "win32" else [])
    if not any(str(resolved).startswith(str(r)) for r in safe_roots):
        raise ValueError(f"Unsafe path: {resolved}")
    return resolved


def calculate_optimal_chunk_size(file_size: int, threads: int,
                                 min_chunk: int = 1 * 1024 * 1024) -> int:
    chunk = file_size // threads
    return max(chunk, min_chunk)
