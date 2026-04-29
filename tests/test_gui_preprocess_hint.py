import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.main import MainWindow


def test_preprocess_recommendation_label_shows_after_load():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    # Create a small df with a text and date column to trigger recommendations
    import pandas as pd
    df = pd.DataFrame({
        "event_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "comment": ["good product", "bad", "ok", "excellent"],
        "target": [0,1,0,1],
    })

    # Inject into UI
    win.current_df = df
    win.csv_path = Path("/tmp/fake.csv")
    win.target_column = "target"

    # Force data quality refresh
    win._refresh_data_quality()

    # The label should be populated with a short recommendation
    text = win.lbl_preprocess_recs.text()
    assert text != ""
