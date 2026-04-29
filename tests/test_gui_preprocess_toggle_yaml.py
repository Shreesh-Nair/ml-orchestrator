import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.main import MainWindow


def test_toggles_control_yaml_generation(tmp_path):
    app = QApplication.instance() or QApplication([])
    win = MainWindow()

    # Minimal CSV path (not read during YAML write)
    win.csv_path = Path(tmp_path) / "dummy.csv"
    win.training_csv_path = None
    win.target_column = "target"

    # Provide auto-recommendations
    recs = {
        "date_extract": True,
        "text_extract": True,
        "text_feature_columns": ["comment"],
        "rare_category_min_freq": 0.05,
    }
    win._auto_preprocess_recommendations = recs

    # Make sure toggles exist and are checked -> should include all
    win.chk_rec_date_extract.setVisible(True)
    win.chk_rec_date_extract.setChecked(True)
    win.chk_rec_text_extract.setVisible(True)
    win.chk_rec_text_extract.setChecked(True)
    win.chk_rec_rare_grouping.setVisible(True)
    win.chk_rec_rare_grouping.setChecked(True)

    yaml_path = win._write_generated_yaml()
    assert yaml_path.exists()
    txt = yaml_path.read_text(encoding="utf-8")
    assert "date_extract" in txt
    assert "text_extract" in txt
    assert "text_feature_columns" in txt
    assert "rare_category_min_freq" in txt

    # Now disable text extraction toggle and regenerate
    win.chk_rec_text_extract.setChecked(False)
    yaml_path2 = win._write_generated_yaml()
    txt2 = yaml_path2.read_text(encoding="utf-8")
    assert "date_extract" in txt2
    assert "text_extract" not in txt2
