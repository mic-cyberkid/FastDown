# liquidglass_downloader/core/config.py
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AppConfig:
    """Validated configuration model."""

    def __init__(self, **kwargs: Any):
        self.threads = kwargs.get("threads", 8)
        self.chunk_size = kwargs.get("chunk_size", 8192)
        self.theme = kwargs.get("theme", "dark")
        self.window_size = kwargs.get("window_size", [1000, 700])
        self.download_folder = kwargs.get("download_folder",
                                         str(Path.home() / "Downloads"))
        self.auto_verify_hash = kwargs.get("auto_verify_hash", True)
        self.high_priority = kwargs.get("high_priority", True)
        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 300)
        self.max_bandwidth = kwargs.get("max_bandwidth", None)

        self._validate_and_normalize()

    # ------------------------------------------------------------------ #
    # (validation logic unchanged – omitted for brevity)
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def copy(self, **updates: Any) -> "AppConfig":
        data = self.to_dict()
        data.update(updates)
        return AppConfig(**data)


class ValidatedConfigManager:
    """Persists and validates AppConfig."""

    CONFIG_FILE = Path("downloader_config.json")

    def __init__(self) -> None:
        self.config: AppConfig = self._load()

    def _load(self) -> AppConfig:
        default = AppConfig()
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                merged = {**default.to_dict(), **loaded}
                return AppConfig(**merged)
            except Exception as e:
                logger.warning("Config load failed – using defaults: %s", e)
        return default

    def save(self) -> None:
        try:
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any) -> None:
        if not hasattr(self.config, key):
            raise AttributeError(key)
        new_cfg = self.config.copy(**{key: value})
        self.config = new_cfg
        self.save()
