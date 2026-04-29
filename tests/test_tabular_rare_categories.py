import pandas as pd

from core.yaml_parser import Stage
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler


def test_rare_categories_are_grouped_into_other_bucket():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "A", "A", "B", "C", "D", None, "A", "A"],
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        }
    )

    stage = Stage(
        name="preprocess",
        type="tabular_preprocess",
        params={
            "task_type": "classification",
            "require_binary_target": True,
            "encode_categoricals": True,
            "rare_category_min_freq": 0.2,
            "test_size": 0.4,
            "random_state": 42,
        },
    )
    handler = TabularPreprocessHandler(stage)

    out = handler.run({"df": df, "target_column": "target"})

    preprocessor = out["preprocessor"]
    cat_transformer = preprocessor.named_transformers_["cat"]
    categories = cat_transformer.named_steps["onehot"].categories_[0].tolist()

    assert "__OTHER__" in categories
    assert "C" not in categories
    assert "D" not in categories
