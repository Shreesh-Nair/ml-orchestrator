import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from gui.main import MainWindow


def test_gui_launch_and_quit_headless():
    """Smoke test: launch the main window headlessly and quit after 1s."""
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()

    # Quit the app after 1 second to keep test short
    QTimer.singleShot(1000, app.quit)
    app.exec()

    # Basic assertion that the window object was created
    assert win is not None
