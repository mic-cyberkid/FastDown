# core/engine.py
import hashlib
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict, Any

import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ValidatedConfigManager
from .state import DownloadStateManager
from .utils import calculate_optimal_chunk_size

logger = logging.getLogger(__name__)


class DownloadEngine:
    def __init__(self, cfg_mgr: ValidatedConfigManager):
        self.cfg = cfg_mgr
        self.session: Optional[requests.Session] = None
        self.is_downloading = False
        self.state_manager: Optional[DownloadStateManager] = None

    # ------------------------------------------------------------------ #
    # Session, priority, file-info, temp-dir, state init (unchanged)
    # ------------------------------------------------------------------ #

    def _setup_session(self) -> requests.Session:
        sess = requests.Session()
        retry = Retry(total=self.cfg.get("max_retries", 3),
                      backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "HEAD"])
        adapter = HTTPAdapter(pool_connections=100,
                              pool_maxsize=100,
                              max_retries=retry)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        sess.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        return sess

    # … (get_file_info, generate_temp_dir, initialize_state_manager,
    #      download_range, download_parts, combine_parts, verify_hash,
    #      set_high_priority, stop_download) …

    def download_file(
        self,
        url: str,
        output_path: str,
        num_threads: int = 8,
        expected_hash: Optional[str] = None,
        high_priority: bool = False,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        """Public entry point – identical API as the original monolith."""
        self.is_downloading = True
        self.session = self.session or self._setup_session()
        if high_priority:
            self._set_high_priority()

        try:
            if status_callback:
                status_callback("Fetching file info…")
            info = self.get_file_info(url)
            file_size = info["size"]
            if file_size <= 0:
                raise ValueError("File size unknown")

            temp_dir = self.generate_temp_dir(url, output_path)
            self.state_manager = self.initialize_state_manager(
                url, output_path, file_size, num_threads
            )

            ranges = self._build_ranges(file_size, num_threads)
            if status_callback:
                status_callback("Downloading parts…")
            completed = self.download_parts(url, ranges, temp_dir,
                                            progress_callback)

            if not self.is_downloading:
                return False, "Cancelled"

            if all(completed):
                if status_callback:
                    status_callback("Merging parts…")
                if not self.combine_parts(num_threads, file_size,
                                          output_path, temp_dir):
                    return False, "Merge failed"

                if expected_hash and status_callback:
                    status_callback("Verifying hash…")
                if expected_hash and not self.verify_hash(output_path,
                                                          expected_hash):
                    return False, "Hash mismatch"

                self.state_manager.cleanup()
                shutil.rmtree(temp_dir, ignore_errors=True)
                return True, "Download completed"
            return False, "Some parts failed"
        except Exception as exc:
            logger.exception("Download error")
            return False, str(exc)
        finally:
            self.is_downloading = False
