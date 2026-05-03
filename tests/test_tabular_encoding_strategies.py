import pandas as pd

from core.yaml_parser import Stage
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler


def _run_preprocess(encoding_strategy: str):
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C", "B", "A"],
            "age": [21, 35, 29, 40, 23, 31],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    stage = Stage(
        name="preprocess",
        type="tabular_preprocess",
        params={
            "task_type": "classification",
            "require_binary_target": True,
            "encode_categoricals": True,
            "encoding_strategy": encoding_strategy,
            "test_size": 0.33,
            "random_state": 7,
        },
    )

    return TabularPreprocessHandler(stage).run({"df": df, "target_column": "target"})


def test_onehot_encoding_creates_expanded_columns():
    out = _run_preprocess("onehot")

    preprocessor = out["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    assert "onehot" in cat_transformer.named_steps
    assert out["X_train"].shape[1] > 2


def test_ordinal_encoding_keeps_single_categorical_column():
    out = _run_preprocess("ordinal")

    preprocessor = out["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    assert "ordinal" in cat_transformer.named_steps
    assert out["X_train"].shape[1] == 2


def test_target_encoding_uses_sklearn_target_encoder():
    out = _run_preprocess("target")

    preprocessor = out["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    assert "target" in cat_transformer.named_steps
    assert out["X_train"].shape[1] == 2