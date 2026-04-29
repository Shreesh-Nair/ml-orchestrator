import pandas as pd
from core.yaml_parser import Stage
from handlers.preprocess.tabular_preprocess import TabularPreprocessHandler


def test_date_extraction_adds_columns():
    df = pd.DataFrame({
        "date_str": ["2020-01-05", "2020-02-06", "2020-03-07", "2020-04-08", None],
        "value": [1.0, 2.5, 3.0, 4.2, 5.1],
        "target": [0, 1, 0, 1, 1],
    })

    stage = Stage(name="preprocess", type="tabular_preprocess", params={"date_extract": True, "test_size": 0.4})
    handler = TabularPreprocessHandler(stage)

    context = {"df": df, "target_column": "target"}
    out = handler.run(context)

    feature_cols = out.get("feature_columns", [])
    # Expect year/month/day/hour/weekday extracted for date_str
    assert any(col.startswith("date_str__year") for col in feature_cols)
    assert any(col.startswith("date_str__month") for col in feature_cols)
    assert "value" in feature_cols
