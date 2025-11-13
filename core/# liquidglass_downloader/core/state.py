# liquidglass_downloader/core/state.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import portalocker
import logging

logger = logging.getLogger(__name__)


class DownloadState:
    def __init__(self, url: str, output_path: str, file_size: int,
                 total_parts: int, parts: Optional[Dict] = None,
                 completed: bool = False):
        self.url = url
        self.output_path = output_path
        self.file_size = file_size
        self.total_parts = total_parts
        self.parts = parts or {}
        self.completed = completed
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadState":
        obj = cls(
            url=data["url"],
            output_path=data["output_path"],
            file_size=data["file_size"],
            total_parts=data["total_parts"],
            parts=data.get("parts", {}),
            completed=data.get("completed", False),
        )
        obj.created_at = data.get("created_at", obj.created_at)
        obj.updated_at = data.get("updated_at", obj.updated_at)
        return obj


class DownloadStateManager:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Optional[DownloadState] = None

    def load(self) -> Optional[DownloadState]:
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                portalocker.lock(f, portalocker.LOCK_SH)
                data = json.load(f)
                portalocker.unlock(f)
            self.state = DownloadState.from_dict(data)
            return self.state
        except Exception as e:
            logger.error("State load error: %s", e)
            return None

    def save(self, state: DownloadState) -> None:
        state.updated_at = datetime.now().isoformat()
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(state.to_dict(), f, indent=2)
                portalocker.unlock(f)
            self.state = state
        except Exception as e:
            logger.error("State save error: %s", e)

    def update_part(self, part_num: int, cur: int, tot: int) -> None:
        if self.state:
            self.state.parts[part_num] = {
                "current": cur, "total": tot,
                "timestamp": datetime.now().isoformat()
            }
            self.save(self.state)

    def cleanup(self) -> None:
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception as e:
                logger.warning("State cleanup failed: %s", e)
