# liquidglass_downloader/gui/styles.py
from PyQt5.QtWidgets import QWidget
from ..core.config import ValidatedConfigManager


def load_stylesheet() -> str:
    # The giant CSS block from the original file – unchanged.
    # (Paste the whole `style = """ ... """` string here.)
    return  """
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


def apply_theme(widget: QWidget, cfg: ValidatedConfigManager) -> None:
    # Placeholder – you can expand with light/dark palettes later.
    widget.setStyleSheet(load_stylesheet())
