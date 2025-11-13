# fastdwon/gui/widgets.py
from PyQt5.QtWidgets import QFrame
from PyQt5.QtGui import QPainter, QLinearGradient, QBrush, QColor
from PyQt5.QtCore import Qt


class GlassFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.blur_radius = 15
        self.opacity = 0.9

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(255, 255, 255, int(40 * self.opacity)))
        grad.setColorAt(1, QColor(255, 255, 255, int(20 * self.opacity)))

        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 15, 15)
