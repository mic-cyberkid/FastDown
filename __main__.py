# fastdown/__main__.py
import sys
from .cli import cli_main
from .gui.main_window import LiquidGlassDownloader
from PyQt5.QtWidgets import QApplication


def main() -> None:
    if len(sys.argv) > 1:
        cli_main()                     # CLI mode
    else:
        # GUI mode
        app = QApplication(sys.argv)
        app.setApplicationName("Liquid Glass Downloader")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("LiquidGlass")

        window = LiquidGlassDownloader()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
