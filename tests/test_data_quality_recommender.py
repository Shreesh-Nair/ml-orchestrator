import pandas as pd

from core.data_quality import analyze_data_quality, recommend_quick_fixes


def test_recommender_detects_duplicates_and_constants():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": ["x", "x", "y", "y"],
            "c": [None, None, None, None],
        }
    )
    report = analyze_data_quality(df, target_column=None)
    rec = recommend_quick_fixes(report)
    assert rec["drop_duplicate_rows"] is True
    assert rec["drop_constant_columns"] is True


def test_recommender_handles_missingness_small_fill():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", "z", "w"]})
    report = analyze_data_quality(df)
    rec = recommend_quick_fixes(report)
    assert rec["missing_strategy"] in {"fill_simple", "drop_rows"}


def test_recommender_high_missing_drops_rows():
    df = pd.DataFrame({"a": [1, None, None, None], "b": [1, 2, 3, 4]})
    report = analyze_data_quality(df)
    rec = recommend_quick_fixes(report)
    assert rec["missing_strategy"] == "drop_rows"
