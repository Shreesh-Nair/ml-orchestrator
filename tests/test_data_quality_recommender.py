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


def test_recommender_adds_date_and_text_preprocess_params():
    df = pd.DataFrame(
        {
            "event_date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "review": [
                "this product works really well",
                "bad quality and very slow",
                "excellent support from team",
                "okay but could be better",
                "solid value for the money",
            ],
            "target": [0, 1, 0, 1, 0],
        }
    )

    report = analyze_data_quality(df, target_column="target")
    rec = recommend_quick_fixes(report)
    params = rec.get("preprocess_params") or {}

    assert params.get("date_extract") is True
    assert params.get("text_extract") is True
    assert "review" in params.get("text_feature_columns", [])


def test_recommender_adds_rare_category_grouping_param():
    city_values = ["hub"] * 60 + [f"rare_{i}" for i in range(40)]
    df = pd.DataFrame(
        {
            "city": city_values,
            "target": [0 if i % 2 == 0 else 1 for i in range(len(city_values))],
        }
    )

    report = analyze_data_quality(df, target_column="target")
    rec = recommend_quick_fixes(report)
    params = rec.get("preprocess_params") or {}

    assert params.get("rare_category_min_freq") == 0.05
