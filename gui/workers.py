# liquidglass_downloader/gui/workers.py
from PyQt5.QtCore import QThread, pyqtSignal
from ..core.engine import DownloadEngine
from typing import Optional, Callable


class FileInfoWorker(QThread):
    info_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, url: str, engine: DownloadEngine):
        super().__init__()
        self.url = url
        self.engine = engine

    def run(self) -> None:
        try:
            self.info_ready.emit(self.engine.get_file_info(self.url))
        except Exception as e:
            self.error_occurred.emit(str(e))


class DownloadWorker(QThread):
    progress_updated = pyqtSignal(int, int, int)      # part, cur, tot
    overall_progress = pyqtSignal(float, str)        # % , status
    download_finished = pyqtSignal(bool, str)       # success, msg
    status_updated = pyqtSignal(str)

    def __init__(self, engine: DownloadEngine):
        super().__init__()
        self.engine = engine
        self.params: dict = {}

    def start_download(self, **kwargs) -> None:
        self.params = kwargs
        self.start()

    def run(self) -> None:
        def prog(part, cur, tot):
            self.progress_updated.emit(part, cur, tot)

        def stat(msg):
            self.status_updated.emit(msg)

        success, msg = self.engine.download_file(
            progress_callback=prog,
            status_callback=stat,
            **self.params,
        )
        self.download_finished.emit(success, msg)
