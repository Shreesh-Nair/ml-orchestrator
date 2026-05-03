import pandas as pd

from core.yaml_parser import Stage
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler


def _run_preprocess(scale_strategy: str | None = None, scale_numeric: bool | None = None):
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C", "B", "A"],
            "age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    params = {
        "task_type": "classification",
        "require_binary_target": True,
        "encode_categoricals": True,
        "encoding_strategy": "onehot",
        "test_size": 0.33,
        "random_state": 11,
    }
    if scale_strategy is not None:
        params["scale_strategy"] = scale_strategy
    if scale_numeric is not None:
        params["scale_numeric"] = scale_numeric

    stage = Stage(name="preprocess", type="tabular_preprocess", params=params)
    return TabularPreprocessHandler(stage).run({"df": df, "target_column": "target"})


def test_standard_scaling_uses_standard_scaler():
    out = _run_preprocess(scale_strategy="standard")
    num_transformer = out["preprocessor"].named_transformers_["num"]

    assert "scaler" in num_transformer.named_steps
    assert num_transformer.named_steps["scaler"].__class__.__name__ == "StandardScaler"


def test_minmax_scaling_uses_minmax_scaler():
    out = _run_preprocess(scale_strategy="minmax")
    num_transformer = out["preprocessor"].named_transformers_["num"]

    assert "scaler" in num_transformer.named_steps
    assert num_transformer.named_steps["scaler"].__class__.__name__ == "MinMaxScaler"


def test_none_scaling_skips_scaler_and_keeps_backward_compatibility():
    out = _run_preprocess(scale_strategy="none")
    num_transformer = out["preprocessor"].named_transformers_["num"]

    assert "scaler" not in getattr(num_transformer, "named_steps", {})

    legacy_out = _run_preprocess(scale_numeric=False)
    legacy_num_transformer = legacy_out["preprocessor"].named_transformers_["num"]

    assert "scaler" not in getattr(legacy_num_transformer, "named_steps", {})