import requests
import threading
import os
import psutil
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import logging
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import portalocker
import time
import json
from datetime import datetime
from pathlib import Path
import tempfile
from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import multiprocessing

# Qt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QProgressBar, QFrame, 
                             QGroupBox, QComboBox, QCheckBox, QFileDialog, QMessageBox,
                             QScrollArea, QGridLayout, QSplitter, QTextEdit, QTabWidget,
                             QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
from PyQt5.QtCore import pyqtProperty

# Configure enhanced logging
def setup_logging():
    """Setup enhanced logging with JSON-like format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler("downloader.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration Models (Pydantic replacement with manual validation)
class AppConfig:
    """Application configuration model with validation."""
    
    def __init__(self, **kwargs):
        self.threads = kwargs.get('threads', 8)
        self.chunk_size = kwargs.get('chunk_size', 8192)
        self.theme = kwargs.get('theme', 'dark')
        self.window_size = kwargs.get('window_size', [1000, 700])
        self.download_folder = kwargs.get('download_folder', str(Path.home() / "Downloads"))
        self.auto_verify_hash = kwargs.get('auto_verify_hash', True)
        self.high_priority = kwargs.get('high_priority', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout = kwargs.get('timeout', 300)
        self.max_bandwidth = kwargs.get('max_bandwidth', None)
        
        # Validate and normalize
        self._validate_and_normalize()
    
    def _validate_and_normalize(self):
        """Validate configuration values and set defaults."""
        # Thread validation
        max_recommended = multiprocessing.cpu_count() * 4
        if self.threads > max_recommended:
            logger.warning(f"Thread count {self.threads} exceeds recommended maximum {max_recommended}")
            self.threads = min(self.threads, max_recommended)
        
        # Chunk size validation
        self.chunk_size = max(1024, min(self.chunk_size, 65536))
        
        # Theme validation
        if self.theme not in ['dark', 'light', 'system']:
            self.theme = 'dark'
        
        # Timeout validation
        self.timeout = max(30, min(self.timeout, 600))
        
        # Download folder validation
        path = Path(self.download_folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        self.download_folder = str(path.resolve())
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'threads': self.threads,
            'chunk_size': self.chunk_size,
            'theme': self.theme,
            'window_size': self.window_size,
            'download_folder': self.download_folder,
            'auto_verify_hash': self.auto_verify_hash,
            'high_priority': self.high_priority,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'max_bandwidth': self.max_bandwidth
        }
    
    def copy(self, **updates):
        """Create a copy with updated values."""
        config_dict = self.to_dict()
        config_dict.update(updates)
        return AppConfig(**config_dict)

class DownloadState:
    """Download state for resuming interrupted downloads."""
    
    def __init__(self, url: str, output_path: str, file_size: int, total_parts: int, 
                 parts: Optional[Dict] = None, completed: bool = False):
        self.url = url
        self.output_path = output_path
        self.file_size = file_size
        self.total_parts = total_parts
        self.parts = parts or {}
        self.completed = completed
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert state to dictionary."""
        return {
            'url': self.url,
            'output_path': self.output_path,
            'file_size': self.file_size,
            'total_parts': self.total_parts,
            'parts': self.parts,
            'completed': self.completed,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create state from dictionary."""
        state = cls(
            url=data['url'],
            output_path=data['output_path'],
            file_size=data['file_size'],
            total_parts=data['total_parts'],
            parts=data.get('parts', {}),
            completed=data.get('completed', False)
        )
        state.created_at = data.get('created_at', state.created_at)
        state.updated_at = data.get('updated_at', state.updated_at)
        return state

class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Utility Functions
def safe_output_path(user_path: str, default_dir: Optional[Path] = None) -> Path:
    """
    Convert user input to safe absolute path.
    
    Args:
        user_path: User-provided path
        default_dir: Default directory if path is relative
    
    Returns:
        Safe absolute Path object
    """
    path = Path(user_path)
    if not path.is_absolute():
        path = (default_dir or Path.home() / "Downloads") / path
    
    # Resolve to absolute path and ensure it's within user directories
    resolved = path.resolve()
    
    # Security check: ensure path doesn't escape intended directories
    safe_dirs = [Path.home(), Path("/tmp")] if sys.platform != "win32" else [Path.home()]
    if not any(str(resolved).startswith(str(safe_dir)) for safe_dir in safe_dirs):
        raise ValueError(f"Path {resolved} is not in a safe directory")
    
    return resolved

def calculate_optimal_chunk_size(file_size: int, num_threads: int, min_chunk: int = 1*1024*1024) -> int:
    """Calculate optimal chunk size based on file size and thread count."""
    chunk = file_size // num_threads
    return max(chunk, min_chunk)

class DownloadStateManager:
    """Manage download state for resuming interrupted downloads."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Optional[DownloadState] = None
        self._lock = threading.RLock()
    
    def load_state(self) -> Optional[DownloadState]:
        """Load download state from file."""
        with self._lock:
            try:
                if self.state_file.exists():
                    # Use file locking for concurrent access safety
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        portalocker.lock(f, portalocker.LOCK_SH)
                        data = json.load(f)
                        portalocker.unlock(f)
                        
                    self.state = DownloadState.from_dict(data)
                    logger.info("Download state loaded from %s", self.state_file)
                    return self.state
            except Exception as e:
                logger.error("Failed to load download state from %s: %s", self.state_file, e)
            return None
    
    def save_state(self, state: DownloadState):
        """Save download state to file."""
        with self._lock:
            try:
                state.updated_at = datetime.now().isoformat()
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    portalocker.lock(f, portalocker.LOCK_EX)
                    json.dump(state.to_dict(), f, indent=2, default=str)
                    portalocker.unlock(f)
                
                self.state = state
                logger.debug("Download state saved to %s", self.state_file)
            except Exception as e:
                logger.error("Failed to save download state to %s: %s", self.state_file, e)
    
    def update_part_progress(self, part_num: int, current_size: int, total_size: int):
        """Update progress for a specific part."""
        if self.state:
            self.state.parts[part_num] = {
                'current': current_size,
                'total': total_size,
                'timestamp': datetime.now().isoformat()
            }
            self.save_state(self.state)
    
    def cleanup(self):
        """Remove state file when download is complete."""
        with self._lock:
            try:
                if self.state_file.exists():
                    self.state_file.unlink()
                    logger.info("Download state cleaned up: %s", self.state_file)
            except Exception as e:
                logger.warning("Failed to cleanup state file %s: %s", self.state_file, e)

class ValidatedConfigManager:
    """Manage application configuration with validation."""
    
    def __init__(self):
        self.config_path = Path("downloader_config.json")
        self.config: AppConfig = self.load_config()
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        default_config = AppConfig()
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    config_dict = {**default_config.to_dict(), **loaded_config}
                    return AppConfig(**config_dict)
            return default_config
        except Exception as e:
            logger.warning("Failed to load config, using defaults: %s", e)
            return default_config
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, default=str)
            logger.debug("Configuration saved to %s", self.config_path)
        except Exception as e:
            logger.error("Failed to save config to %s: %s", self.config_path, e)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return getattr(self.config, key, default)
    
    def set(self, key: str, value):
        """Set configuration value with validation."""
        if hasattr(self.config, key):
            # Create new config with updated value
            updates = {key: value}
            new_config = self.config.copy(**updates)
            self.config = new_config
            self.save_config()
        else:
            raise AttributeError(f"Invalid config key: {key}")

class DownloadEngine:
    """Core download engine used by both CLI and GUI."""
    
    def __init__(self, config_manager: ValidatedConfigManager):
        self.config = config_manager
        self.session: Optional[requests.Session] = None
        self.is_downloading = False
        self.state_manager: Optional[DownloadStateManager] = None
    
    def setup_session(self) -> requests.Session:
        """Create a requests session with retry mechanism and connection pooling."""
        session = requests.Session()
        retries = Retry(
            total=self.config.get('max_retries', 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET', 'HEAD']
        )
        
        adapter = HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=retries
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        return session
    
    def set_high_priority(self):
        """Set the process to high priority."""
        try:
            if sys.platform == "win32":
                p = psutil.Process(os.getpid())
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            elif sys.platform in ["linux", "darwin"]:
                os.nice(-10)
            logger.info("Set process to high priority")
        except Exception as e:
            logger.warning("Failed to set high priority: %s", e)
    
    def get_file_info(self, url: str) -> Dict[str, Any]:
        """Get file information from URL."""
        if not self.session:
            self.session = self.setup_session()
        
        try:
            response = self.session.head(url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                content_type = response.headers.get('content-type', 'Unknown')
                filename = os.path.basename(urlparse(url).path) or "download.file"
                
                # Check if server supports range requests
                accept_ranges = response.headers.get('accept-ranges', '').lower() == 'bytes'
                supports_ranges = accept_ranges or size > 0
                
                return {
                    'size': size,
                    'type': content_type,
                    'filename': filename,
                    'supported': supports_ranges and size > 0,
                    'accept_ranges': accept_ranges
                }
            else:
                raise ValueError(f"HTTP {response.status_code}")
        except Exception as e:
            logger.error("Failed to get file info for %s: %s", url, e)
            raise
    
    def generate_temp_dir(self, url: str, output_file: str) -> str:
        """Generate a unique temp directory for download parts."""
        hash_input = f"{url}:{output_file}"
        dir_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"temp_parts_{dir_hash}"
    
    def initialize_state_manager(self, url: str, output_path: str, file_size: int, num_threads: int) -> DownloadStateManager:
        """Initialize state manager for resume capability."""
        state_file = Path(f"download_state_{hashlib.md5(url.encode()).hexdigest()[:8]}.json")
        state_manager = DownloadStateManager(state_file)
        
        # Load existing state or create new
        existing_state = state_manager.load_state()
        if not existing_state:
            new_state = DownloadState(
                url=url,
                output_path=output_path,
                file_size=file_size,
                total_parts=num_threads
            )
            state_manager.save_state(new_state)
        
        return state_manager
    
    def download_range(self, url: str, start: int, end: int, part_num: int, 
                      output_dir: str, progress_callback: Optional[Callable] = None) -> bool:
        """Download a specific range of the file with resume support."""
        part_file = os.path.join(output_dir, f"part_{part_num}.bin")
        total_size = end - start + 1
        
        # Check existing part for resume
        current_size = 0
        if os.path.exists(part_file):
            current_size = os.path.getsize(part_file)
            if current_size >= total_size:
                logger.debug("Part %d already completed: %d bytes", part_num, current_size)
                if progress_callback:
                    progress_callback(part_num, total_size, total_size)
                return True
        
        headers = {'Range': f'bytes={start + current_size}-{end}'}
        mode = 'ab' if current_size > 0 else 'wb'
        
        try:
            with self.session.get(url, headers=headers, stream=True, timeout=60) as response:
                if response.status_code not in [206, 200]:
                    logger.error("Failed to download part %d: HTTP %d", part_num, response.status_code)
                    return False
                
                os.makedirs(output_dir, exist_ok=True)
                with open(part_file, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.config.get('chunk_size', 8192)):
                        if not self.is_downloading:
                            return False
                        if chunk:
                            f.write(chunk)
                            current_size += len(chunk)
                            
                            # Update progress
                            if progress_callback:
                                progress_callback(part_num, current_size, total_size)
                            
                            # Update state for resume
                            if self.state_manager:
                                self.state_manager.update_part_progress(part_num, current_size, total_size)
                            
                            f.flush()
                            os.fsync(f.fileno())
                
                final_size = os.path.getsize(part_file)
                success = final_size == total_size
                if success:
                    logger.info("Part %d downloaded successfully: %d bytes", part_num, final_size)
                else:
                    logger.warning("Part %d incomplete: %d/%d bytes", part_num, final_size, total_size)
                return success
                
        except Exception as e:
            logger.error("Error downloading part %d: %s", part_num, e)
            return False
    
    def download_parts(self, url: str, ranges: List[Tuple[int, int]], temp_dir: str,
                      progress_callback: Optional[Callable] = None) -> List[bool]:
        """Download all parts with proper resource management."""
        completed_parts = [False] * len(ranges)
        
        with ExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=len(ranges)))
            
            # Submit all tasks
            futures = {
                executor.submit(self.download_range, url, start, end, i, temp_dir, progress_callback): i
                for i, (start, end) in enumerate(ranges)
            }
            
            # Process completed tasks
            for future in as_completed(futures):
                if not self.is_downloading:
                    # Cancel remaining tasks
                    for f in futures:
                        f.cancel()
                    break
                
                part_num = futures[future]
                try:
                    completed_parts[part_num] = future.result(timeout=self.config.get('timeout', 300))
                except Exception as e:
                    logger.error("Part %d download failed: %s", part_num, e)
                    completed_parts[part_num] = False
                    
        return completed_parts
    
    def combine_parts(self, total_parts: int, file_size: int, output_file: str, temp_dir: str) -> bool:
        """Combine downloaded parts into final file with verification."""
        try:
            # Verify all parts exist and have correct sizes
            for i in range(total_parts):
                part_file = os.path.join(temp_dir, f"part_{i}.bin")
                if not os.path.exists(part_file):
                    logger.error("Part file missing: %d", i)
                    return False
            
            # Combine parts
            with open(output_file, 'wb') as outfile:
                for i in range(total_parts):
                    part_file = os.path.join(temp_dir, f"part_{i}.bin")
                    with open(part_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    # Remove part file after successful combination
                    os.remove(part_file)
            
            # Verify final file size
            final_size = os.path.getsize(output_file)
            if final_size != file_size:
                logger.error("File size mismatch after combination: expected %d, got %d", file_size, final_size)
                return False
            
            logger.info("Parts combined successfully: %s (%d bytes)", output_file, final_size)
            return True
            
        except Exception as e:
            logger.error("Error combining parts: %s", e)
            return False
    
    def verify_hash(self, output_file: str, expected_hash: Optional[str]) -> bool:
        """Verify file hash with multiple algorithms."""
        if not expected_hash:
            return True
        
        try:
            hash_type, expected_value = expected_hash.lower().split(':', 1)
            hash_func = {
                'md5': hashlib.md5,
                'sha1': hashlib.sha1,
                'sha256': hashlib.sha256,
                'sha512': hashlib.sha512
            }.get(hash_type)
            
            if not hash_func:
                raise ValueError(f"Unsupported hash type: {hash_type}")
            
            hasher = hash_func()
            with open(output_file, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            computed = hasher.hexdigest()
            if computed == expected_value:
                logger.info("Hash verification successful: %s", hash_type.upper())
                return True
            else:
                logger.error("Hash mismatch: %s (computed: %s, expected: %s)", 
                           hash_type.upper(), computed, expected_value)
                return False
                
        except Exception as e:
            logger.error("Hash verification failed: %s", e)
            return False
    
    def download_file(self, url: str, output_path: str, num_threads: int = 8,
                     expected_hash: Optional[str] = None, high_priority: bool = False,
                     progress_callback: Optional[Callable] = None,
                     status_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Main download method used by both CLI and GUI interfaces."""
        self.is_downloading = True
        
        try:
            # Setup
            if not self.session:
                self.session = self.setup_session()
            
            if high_priority:
                self.set_high_priority()
            
            # Get file info
            if status_callback:
                status_callback("Getting file information...")
            
            file_info = self.get_file_info(url)
            file_size = file_info['size']
            
            if file_size == 0:
                raise ValueError("File size is 0 or cannot be determined")
            
            if status_callback:
                status_callback(f"File size: {file_size} bytes")
            
            # Setup temp directory and state management
            temp_dir = self.generate_temp_dir(url, output_path)
            self.state_manager = self.initialize_state_manager(url, output_path, file_size, num_threads)
            
            # Calculate optimal ranges
            chunk_size = calculate_optimal_chunk_size(file_size, num_threads)
            ranges = []
            for i in range(num_threads):
                start = i * chunk_size
                end = start + chunk_size - 1 if i < num_threads - 1 else file_size - 1
                ranges.append((start, end))
            
            # Download parts
            if status_callback:
                status_callback("Starting parallel download...")
            
            completed_parts = self.download_parts(url, ranges, temp_dir, progress_callback)
            
            if not self.is_downloading:
                return False, "Download cancelled"
            
            if all(completed_parts):
                # Combine parts
                if status_callback:
                    status_callback("Combining parts...")
                
                if self.combine_parts(num_threads, file_size, output_path, temp_dir):
                    # Verify hash
                    if expected_hash:
                        if status_callback:
                            status_callback("Verifying hash...")
                        
                        if self.verify_hash(output_path, expected_hash):
                            # Cleanup
                            self.state_manager.cleanup()
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            return True, "Download completed and verified"
                        else:
                            return False, "Hash verification failed"
                    else:
                        # Cleanup
                        self.state_manager.cleanup()
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return True, "Download completed successfully"
                else:
                    return False, "Failed to combine parts"
            else:
                return False, "Download incomplete - some parts failed"
                
        except Exception as e:
            logger.error("Download error: %s", e)
            return False, f"Download failed: {str(e)}"
        finally:
            self.is_downloading = False
    
    def stop_download(self):
        """Stop the current download."""
        self.is_downloading = False

# Qt Worker Threads
class FileInfoWorker(QThread):
    """Worker thread for fetching file information."""
    
    info_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, url: str, engine: DownloadEngine):
        super().__init__()
        self.url = url
        self.engine = engine
    
    def run(self):
        try:
            info = self.engine.get_file_info(self.url)
            self.info_ready.emit(info)
        except Exception as e:
            self.error_occurred.emit(str(e))

class DownloadWorker(QThread):
    """Worker thread for handling downloads in GUI."""
    
    progress_updated = pyqtSignal(int, int, int)  # part_num, current, total
    overall_progress = pyqtSignal(float, str)  # progress, status
    download_finished = pyqtSignal(bool, str)  # success, message
    status_updated = pyqtSignal(str)  # status message
    
    def __init__(self, engine: DownloadEngine):
        super().__init__()
        self.engine = engine
        self.download_params = {}
    
    def start_download(self, url: str, output_path: str, num_threads: int,
                      expected_hash: Optional[str] = None, high_priority: bool = False):
        """Set download parameters and start thread."""
        self.download_params = {
            'url': url,
            'output_path': output_path,
            'num_threads': num_threads,
            'expected_hash': expected_hash,
            'high_priority': high_priority
        }
        self.start()
    
    def run(self):
        """Execute download in thread."""
        def progress_callback(part_num: int, current: int, total: int):
            self.progress_updated.emit(part_num, current, total)
        
        def status_callback(status: str):
            self.status_updated.emit(status)
        
        success, message = self.engine.download_file(
            progress_callback=progress_callback,
            status_callback=status_callback,
            **self.download_params
        )
        
        self.download_finished.emit(success, message)

# UI Components (Keeping the original UI implementation)
class GlassFrame(QFrame):
    """A glass-morphism effect frame."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.blur_radius = 15
        self.opacity = 0.9
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(255, 255, 255, int(40 * self.opacity)))
        gradient.setColorAt(1, QColor(255, 255, 255, int(20 * self.opacity)))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

class LiquidGlassDownloader(QMainWindow):
    """Main application window with liquid glass design."""
    
    def __init__(self):
        super().__init__()
        self.config = ValidatedConfigManager()
        self.download_engine = DownloadEngine(self.config)
        self.download_worker = DownloadWorker(self.download_engine)
        self.file_info_worker = None
        self.current_progress = {}
        self.setup_ui()
        self.setup_connections()
        self.apply_theme()
        
    def setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Liquid Glass Downloader - Production Ready")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(900, 600)
        
        # Central widget with glass effect
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        self.setup_header(main_layout)
        
        # Main content area
        self.setup_content_area(main_layout)
        
        # Status bar
        self.setup_status_bar(main_layout)
        
        # Apply styles
        self.apply_styles()
    
    def setup_header(self, layout):
        """Setup the header section."""
        header_frame = GlassFrame()
        header_frame.setFixedHeight(80)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(2)
        
        title_label = QLabel("LIQUID GLASS DOWNLOADER - PRODUCTION")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(30)
        
        subtitle_label = QLabel("High-speed parallel file downloader with resume capability")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFixedHeight(20)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header_frame)
    
    def setup_content_area(self, layout):
        """Setup the main content area with tabs."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("tabWidget")
        
        # Download tab
        download_tab = QWidget()
        self.setup_download_tab(download_tab)
        self.tab_widget.addTab(download_tab, "Download")
        
        # Progress tab
        progress_tab = QWidget()
        self.setup_progress_tab(progress_tab)
        self.tab_widget.addTab(progress_tab, "Progress")
        
        # Settings tab
        settings_tab = QWidget()
        self.setup_settings_tab(settings_tab)
        self.tab_widget.addTab(settings_tab, "Settings")
        
        layout.addWidget(self.tab_widget)
    
    def setup_download_tab(self, parent):
        """Setup the download tab."""
        main_layout = QVBoxLayout(parent)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # URL Section
        url_group = self.create_url_section()
        main_layout.addWidget(url_group)
        
        # Output Section
        output_group = self.create_output_section()
        main_layout.addWidget(output_group)
        
        # Control buttons
        control_group = self.create_control_section()
        main_layout.addWidget(control_group)
        
        main_layout.addStretch(1)
    
    def create_url_section(self):
        """Create URL input section."""
        group = QGroupBox("Download URL")
        group.setObjectName("glassGroup")
        group.setFixedHeight(120)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(10)
        
        # URL input row
        url_layout = QHBoxLayout()
        url_layout.setSpacing(10)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com/file.zip")
        self.url_input.setFixedHeight(35)
        
        self.fetch_btn = QPushButton("Fetch Info")
        self.fetch_btn.setFixedSize(100, 35)
        
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(self.fetch_btn)
        
        # File info label
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setObjectName("fileInfoLabel")
        self.file_info_label.setFixedHeight(20)
        self.file_info_label.setWordWrap(True)
        
        layout.addLayout(url_layout)
        layout.addWidget(self.file_info_label)
        
        return group
    
    def create_output_section(self):
        """Create output settings section."""
        group = QGroupBox("Output Settings")
        group.setObjectName("glassGroup")
        group.setFixedHeight(130)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(10)
        
        # Output path row
        output_layout = QHBoxLayout()
        output_layout.setSpacing(10)
        
        self.output_input = QLineEdit()
        self.output_input.setText(self.config.get('download_folder'))
        self.output_input.setFixedHeight(35)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedSize(80, 35)
        
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(self.browse_btn)
        
        # Hash row
        hash_layout = QHBoxLayout()
        hash_layout.setSpacing(10)
        
        hash_label = QLabel("Expected Hash:")
        hash_label.setFixedWidth(100)
        hash_label.setStyleSheet("color: white;")
        
        self.hash_input = QLineEdit()
        self.hash_input.setPlaceholderText("sha256:abc123...")
        self.hash_input.setFixedHeight(35)
        
        hash_layout.addWidget(hash_label)
        hash_layout.addWidget(self.hash_input)
        
        layout.addLayout(output_layout)
        layout.addLayout(hash_layout)
        
        return group
    
    def create_control_section(self):
        """Create control buttons section."""
        group = QGroupBox()
        group.setObjectName("glassGroup")
        group.setFixedHeight(80)
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.download_btn = QPushButton("Start Download")
        self.download_btn.setObjectName("downloadButton")
        self.download_btn.setFixedSize(120, 40)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pauseButton")
        self.pause_btn.setFixedSize(80, 40)
        self.pause_btn.setEnabled(False)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.setFixedSize(80, 40)
        self.cancel_btn.setEnabled(False)
        
        button_layout.addWidget(self.download_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch(1)
        
        layout.addLayout(button_layout)
        
        return group
    
    def setup_progress_tab(self, parent):
        """Setup the progress monitoring tab."""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Overall progress
        overall_group = QGroupBox("Overall Progress")
        overall_group.setObjectName("glassGroup")
        overall_group.setFixedHeight(100)
        
        overall_layout = QVBoxLayout(overall_group)
        overall_layout.setContentsMargins(15, 20, 15, 15)
        overall_layout.setSpacing(10)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setFixedHeight(20)
        self.overall_status = QLabel("Ready")
        self.overall_status.setFixedHeight(20)
        
        overall_layout.addWidget(self.overall_progress)
        overall_layout.addWidget(self.overall_status)
        
        # Thread progress
        thread_group = QGroupBox("Thread Progress")
        thread_group.setObjectName("glassGroup")
        
        thread_layout = QVBoxLayout(thread_group)
        thread_layout.setContentsMargins(10, 20, 10, 10)
        
        self.thread_scroll_area = QScrollArea()
        self.thread_scroll_area.setWidgetResizable(True)
        self.thread_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thread_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.thread_scroll_widget = QWidget()
        self.thread_scroll_layout = QVBoxLayout(self.thread_scroll_widget)
        self.thread_scroll_layout.setSpacing(5)
        self.thread_scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        self.thread_scroll_area.setWidget(self.thread_scroll_widget)
        
        thread_layout.addWidget(self.thread_scroll_area)
        
        layout.addWidget(overall_group)
        layout.addWidget(thread_group)
    
    def setup_settings_tab(self, parent):
        """Setup the settings tab."""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        settings_group = QGroupBox("Application Settings")
        settings_group.setObjectName("glassGroup")
        settings_group.setFixedHeight(200)
        
        settings_layout = QGridLayout(settings_group)
        settings_layout.setVerticalSpacing(15)
        settings_layout.setHorizontalSpacing(15)
        settings_layout.setContentsMargins(15, 25, 15, 15)
        
        # Threads
        settings_layout.addWidget(QLabel("Threads:"), 0, 0)
        self.threads_combo = QComboBox()
        self.threads_combo.addItems(["1", "2", "4", "8", "16", "32"])
        self.threads_combo.setCurrentText(str(self.config.get('threads', 8)))
        self.threads_combo.setFixedSize(100, 30)
        settings_layout.addWidget(self.threads_combo, 0, 1)
        
        # Theme
        settings_layout.addWidget(QLabel("Theme:"), 1, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.setCurrentText(self.config.get('theme', 'Dark').title())
        self.theme_combo.setFixedSize(100, 30)
        settings_layout.addWidget(self.theme_combo, 1, 1)
        
        # Checkboxes
        self.auto_verify_cb = QCheckBox("Auto verify hash")
        self.auto_verify_cb.setChecked(self.config.get('auto_verify_hash', True))
        settings_layout.addWidget(self.auto_verify_cb, 2, 0, 1, 2)
        
        self.high_priority_cb = QCheckBox("High process priority")
        self.high_priority_cb.setChecked(self.config.get('high_priority', True))
        settings_layout.addWidget(self.high_priority_cb, 3, 0, 1, 2)
        
        # Save button
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.setFixedSize(120, 35)
        settings_layout.addWidget(self.save_settings_btn, 4, 0, 1, 2, Qt.AlignCenter)
        
        layout.addWidget(settings_group)
        layout.addStretch(1)
    
    def setup_status_bar(self, layout):
        """Setup the status bar."""
        status_frame = QFrame()
        status_frame.setFixedHeight(30)
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setFixedHeight(20)
        
        version_label = QLabel("v2.0.0")
        version_label.setObjectName("versionLabel")
        version_label.setFixedHeight(20)
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(version_label)
        
        layout.addWidget(status_frame)
    
    def setup_connections(self):
        """Setup signal-slot connections."""
        # Button connections
        self.fetch_btn.clicked.connect(self.fetch_file_info)
        self.browse_btn.clicked.connect(self.browse_output)
        self.download_btn.clicked.connect(self.toggle_download)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.save_settings_btn.clicked.connect(self.save_settings)
        self.theme_combo.currentTextChanged.connect(self.change_theme)
        
        # Worker connections
        self.download_worker.progress_updated.connect(self.on_progress_updated)
        self.download_worker.overall_progress.connect(self.on_overall_progress)
        self.download_worker.download_finished.connect(self.on_download_finished)
        self.download_worker.status_updated.connect(self.update_status)
    
    def apply_styles(self):
        """Apply CSS styles to the application."""
        style = """
        #centralWidget {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #667eea, stop: 1 #764ba2);
        }
        
        GlassFrame {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }
        
        #glassGroup {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            color: white;
            font-weight: bold;
            margin: 0px;
        }
        
        #glassGroup::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 5px 10px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            color: white;
        }
        
        #titleLabel {
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        
        #subtitleLabel {
            color: rgba(255, 255, 255, 0.8);
            font-size: 12px;
        }
        
        #fileInfoLabel {
            color: #90EE90;
            font-size: 11px;
            padding: 2px 5px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }
        
        #statusLabel {
            color: rgba(255, 255, 255, 0.8);
            font-size: 11px;
        }
        
        #versionLabel {
            color: rgba(255, 255, 255, 0.6);
            font-size: 10px;
        }
        
        QLineEdit {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            padding: 8px;
            color: white;
            font-size: 12px;
        }
        
        QLineEdit:focus {
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        QPushButton {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background: rgba(255, 255, 255, 0.25);
        }
        
        QPushButton:pressed {
            background: rgba(255, 255, 255, 0.35);
        }
        
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        #downloadButton {
            background: rgba(76, 201, 240, 0.7);
        }
        
        #downloadButton:hover {
            background: rgba(76, 201, 240, 0.9);
        }
        
        #pauseButton {
            background: rgba(242, 201, 76, 0.7);
        }
        
        #pauseButton:hover {
            background: rgba(242, 201, 76, 0.9);
        }
        
        #cancelButton {
            background: rgba(242, 92, 84, 0.7);
        }
        
        #cancelButton:hover {
            background: rgba(242, 92, 84, 0.9);
        }
        
        QProgressBar {
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            text-align: center;
            color: white;
            font-size: 11px;
            background: rgba(255, 255, 255, 0.1);
        }
        
        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 #2CC985, stop: 1 #4CC9F0);
            border-radius: 4px;
        }
        
        QTabWidget::pane {
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.05);
        }
        
        QTabBar::tab {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom: none;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            padding: 8px 15px;
            color: white;
            font-weight: bold;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        QTabBar::tab:hover:!selected {
            background: rgba(255, 255, 255, 0.15);
        }
        
        QGroupBox {
            color: white;
            font-weight: bold;
            font-size: 13px;
            margin-top: 10px;
        }
        
        QCheckBox {
            color: white;
            font-size: 12px;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
        }
        
        QCheckBox::indicator:checked {
            background: #4CC9F0;
            border: 1px solid #4CC9F0;
        }
        
        QComboBox {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
            padding: 5px;
            color: white;
            font-size: 12px;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            border: none;
            width: 12px;
            height: 12px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 3px;
        }
        
        QComboBox QAbstractItemView {
            background: rgba(50, 50, 50, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            selection-background-color: #4CC9F0;
        }
        
        QScrollArea {
            border: none;
            background: transparent;
        }
        
        QScrollBar:vertical {
            background: rgba(255, 255, 255, 0.1);
            width: 12px;
            margin: 0px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        QLabel {
            color: white;
        }
        """
        
        self.setStyleSheet(style)
    
    def apply_theme(self):
        """Apply the selected theme."""
        theme = self.theme_combo.currentText().lower()
        self.config.set('theme', theme)
    
    def fetch_file_info(self):
        """Fetch file information from URL using worker thread."""
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a URL")
            return
        
        # Validate URL format
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid URL: {e}")
            return
        
        self.fetch_btn.setEnabled(False)
        self.update_status("Fetching file information...")
        
        # Start worker thread for file info
        self.file_info_worker = FileInfoWorker(url, self.download_engine)
        self.file_info_worker.info_ready.connect(self.on_file_info_ready)
        self.file_info_worker.error_occurred.connect(self.on_file_info_error)
        self.file_info_worker.start()
    
    def on_file_info_ready(self, file_info):
        """Handle file information response."""
        self.fetch_btn.setEnabled(True)
        
        if 'error' in file_info:
            self.file_info_label.setText(f"Error: {file_info['error']}")
            self.file_info_label.setStyleSheet("color: #FF6B6B; background: rgba(255, 107, 107, 0.1); border-radius: 3px;")
            self.update_status(f"Failed to fetch file info: {file_info['error']}", True)
            return
        
        if file_info['supported']:
            size_mb = file_info['size'] / (1024 * 1024)
            supports_ranges = file_info.get('accept_ranges', False)
            range_info = " (Supports resume)" if supports_ranges else " (No resume support)"
            
            self.file_info_label.setText(
                f"{file_info['filename']} - {size_mb:.2f} MB - {file_info['type']}{range_info}"
            )
            self.file_info_label.setStyleSheet("color: #90EE90; background: rgba(144, 238, 144, 0.1); border-radius: 3px;")
            self.download_btn.setEnabled(True)
            self.update_status("File info fetched successfully")
        else:
            self.file_info_label.setText("File size unknown or not supported for parallel download")
            self.file_info_label.setStyleSheet("color: #FF6B6B; background: rgba(255, 107, 107, 0.1); border-radius: 3px;")
            self.download_btn.setEnabled(False)
            self.update_status("File not supported for parallel download", True)
    
    def on_file_info_error(self, error_message):
        """Handle file information error."""
        self.fetch_btn.setEnabled(True)
        self.file_info_label.setText(f"Error: {error_message}")
        self.file_info_label.setStyleSheet("color: #FF6B6B; background: rgba(255, 107, 107, 0.1); border-radius: 3px;")
        self.update_status(f"Failed to fetch file info: {error_message}", True)
    
    def browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Download Directory", self.output_input.text()
        )
        if directory:
            self.output_input.setText(directory)
            self.config.set('download_folder', directory)
    
    def toggle_download(self):
        """Start or resume download with safe path handling."""
        url = self.url_input.text().strip()
        output_path = self.output_input.text().strip()
        expected_hash = self.hash_input.text().strip() if self.auto_verify_cb.isChecked() else None
        
        if not url or not output_path:
            QMessageBox.warning(self, "Error", "Please enter URL and output path")
            return
        
        try:
            # Validate and secure output path
            safe_output = safe_output_path(output_path, Path(self.config.get('download_folder')))
            safe_output.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid output path: {e}")
            return
        
        # Update UI
        self.download_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.fetch_btn.setEnabled(False)
        
        # Switch to progress tab
        self.tab_widget.setCurrentIndex(1)
        
        # Setup progress bars
        self.setup_thread_progress(int(self.threads_combo.currentText()))
        
        # Start download
        self.download_worker.start_download(
            url=url,
            output_path=str(safe_output),
            num_threads=int(self.threads_combo.currentText()),
            expected_hash=expected_hash,
            high_priority=self.high_priority_cb.isChecked()
        )
    
    def toggle_pause(self):
        """Pause or resume download."""
        if self.download_engine.is_downloading:
            self.download_engine.stop_download()
            self.pause_btn.setText("Resume")
            self.update_status("Download paused")
        else:
            self.download_engine.is_downloading = True
            self.pause_btn.setText("Pause")
            # Note: Resume functionality is built into the download engine
            self.update_status("Download resumed")
    
    def cancel_download(self):
        """Cancel the current download."""
        self.download_engine.stop_download()
        self.reset_ui()
        self.update_status("Download cancelled", True)
    
    def setup_thread_progress(self, num_threads):
        """Setup progress bars for individual threads."""
        # Clear existing progress widgets
        for i in reversed(range(self.thread_scroll_layout.count())):
            widget = self.thread_scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.thread_progress_bars = {}
        self.thread_labels = {}
        
        for i in range(num_threads):
            thread_frame = QFrame()
            thread_frame.setFixedHeight(40)
            thread_layout = QHBoxLayout(thread_frame)
            thread_layout.setContentsMargins(10, 5, 10, 5)
            thread_layout.setSpacing(10)
            
            label = QLabel(f"Thread {i+1}:")
            label.setFixedWidth(80)
            label.setStyleSheet("color: white; font-size: 11px;")
            
            progress_bar = QProgressBar()
            progress_bar.setFixedHeight(12)
            
            status_label = QLabel("0%")
            status_label.setFixedWidth(40)
            status_label.setStyleSheet("color: white; font-size: 10px;")
            status_label.setAlignment(Qt.AlignRight)
            
            thread_layout.addWidget(label)
            thread_layout.addWidget(progress_bar)
            thread_layout.addWidget(status_label)
            
            self.thread_scroll_layout.addWidget(thread_frame)
            self.thread_progress_bars[i] = progress_bar
            self.thread_labels[i] = status_label
        
        # Add stretch at the end to push everything to top
        self.thread_scroll_layout.addStretch(1)
    
    def on_progress_updated(self, part_num, current, total):
        """Update progress for a specific part."""
        if part_num in self.thread_progress_bars:
            progress = current / total if total > 0 else 0
            percentage = int(progress * 100)
            
            self.thread_progress_bars[part_num].setValue(percentage)
            self.thread_labels[part_num].setText(f"{percentage}%")
    
    def on_overall_progress(self, progress, status):
        """Update overall progress."""
        percentage = int(progress * 100)
        self.overall_progress.setValue(percentage)
        self.overall_status.setText(f"{percentage}% - {status}")
    
    def on_download_finished(self, success, message):
        """Handle download completion."""
        self.reset_ui()
        
        if success:
            self.overall_progress.setValue(100)
            self.overall_status.setText("100% - Completed!")
            self.update_status("Download completed successfully")
            
            reply = QMessageBox.question(
                self, "Success", 
                f"{message}\n\nOpen containing folder?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    output_dir = os.path.dirname(self.output_input.text())
                    if sys.platform == "win32":
                        os.startfile(output_dir)
                    elif sys.platform == "darwin":
                        os.system(f'open "{output_dir}"')
                    else:
                        os.system(f'xdg-open "{output_dir}"')
                except Exception as e:
                    logger.error("Failed to open folder: %s", e)
        else:
            self.update_status(f"Download failed: {message}", True)
            QMessageBox.critical(self, "Download Error", message)
    
    def reset_ui(self):
        """Reset UI to initial state."""
        self.download_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.cancel_btn.setEnabled(False)
        self.fetch_btn.setEnabled(True)
    
    def update_status(self, message, is_error=False):
        """Update status bar message."""
        color = "#FF6B6B" if is_error else "rgba(255, 255, 255, 0.8)"
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px;")
    
    def save_settings(self):
        """Save application settings."""
        try:
            self.config.set('threads', int(self.threads_combo.currentText()))
            self.config.set('theme', self.theme_combo.currentText().lower())
            self.config.set('auto_verify_hash', self.auto_verify_cb.isChecked())
            self.config.set('high_priority', self.high_priority_cb.isChecked())
            self.config.set('download_folder', self.output_input.text())
            
            self.update_status("Settings saved successfully")
            QMessageBox.information(self, "Success", "Settings saved successfully")
        except Exception as e:
            self.update_status(f"Failed to save settings: {e}", True)
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def change_theme(self, theme):
        """Change application theme."""
        self.apply_theme()
    
    def closeEvent(self, event):
        """Handle application closure."""
        if self.download_engine.is_downloading:
            reply = QMessageBox.question(
                self, "Download in Progress",
                "A download is in progress. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.download_engine.stop_download()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point."""
    # CLI mode
    if len(sys.argv) > 1:
        cli_main()
    else:
        # GUI mode
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("Liquid Glass Downloader")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("LiquidGlass")
        
        # Create and show main window
        window = LiquidGlassDownloader()
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())

def cli_main():
    """CLI mode functionality with improved safety and features."""
    parser = argparse.ArgumentParser(description="Production-ready parallel file downloader with resume and hash check")
    parser.add_argument("url", help="URL of the file to download")
    parser.add_argument("output", help="Output file name")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--expected-hash", help="Expected hash for verification (e.g., md5:abcdef1234567890)")
    parser.add_argument("--high-priority", action="store_true", default=True, help="Set high process priority")
    parser.add_argument("--no-verify", action="store_true", help="Skip hash verification")
    
    args = parser.parse_args()
    
    # Setup
    config = ValidatedConfigManager()
    engine = DownloadEngine(config)
    
    if args.high_priority:
        engine.set_high_priority()
    
    try:
        # Secure output path
        output_path = safe_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def cli_progress(part_num, current, total):
            percent = (current / total) * 100 if total > 0 else 0
            print(f"Part {part_num + 1}: {current}/{total} bytes ({percent:.1f}%)")
        
        def cli_status(status):
            print(f"Status: {status}")
        
        success, message = engine.download_file(
            url=args.url,
            output_path=str(output_path),
            num_threads=args.threads,
            expected_hash=None if args.no_verify else args.expected_hash,
            high_priority=args.high_priority,
            progress_callback=cli_progress,
            status_callback=cli_status
        )
        
        if success:
            print(f"Success: {message}")
            sys.exit(0)
        else:
            print(f"Error: {message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error("CLI download failed: %s", e)
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
