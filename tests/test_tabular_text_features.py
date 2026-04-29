import pandas as pd

from core.yaml_parser import Stage
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler


def test_text_feature_extraction_adds_length_and_word_count():
    df = pd.DataFrame(
        {
            "review": ["great product", "bad", None, "works very well", "ok", "quite good"],
            "city": ["A", "A", "B", "B", "C", "C"],
            "target": [1, 0, 0, 1, 0, 1],
        }
    )

    stage = Stage(
        name="preprocess",
        type="tabular_preprocess",
        params={
            "task_type": "classification",
            "require_binary_target": True,
            "text_extract": True,
            "text_feature_columns": ["review"],
            "text_drop_original": True,
            "test_size": 0.5,
            "random_state": 42,
        },
    )

    out = TabularPreprocessHandler(stage).run({"df": df, "target_column": "target"})
    feature_cols = out["feature_columns"]

    assert "review__char_len" in feature_cols
    assert "review__word_count" in feature_cols
    assert "review" not in feature_cols
