from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from handlers.base import BaseHandler


class TabularPreprocessHandler(BaseHandler):
    """
    Basic tabular preprocessing:
    - Splits df into X (features) and y (target)
    - Imputes missing values
    - One-hot encodes categoricals (if encode_categoricals=True)
    - Scales numerical features (if scale_numeric=True)
    - Train/test split (using test_size param)
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        df = context.get("df")
        target_column = context.get("target_column")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("TabularPreprocessHandler: context['df'] must be a pandas DataFrame")

        if not isinstance(target_column, str) or target_column not in df.columns:
            raise ValueError(
                f"TabularPreprocessHandler: target_column {target_column!r} not found in df.columns"
            )

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        # Read preprocessing params
        impute_missing = bool(self.stage.params.get("impute_missing", True))
        scale_numeric = bool(self.stage.params.get("scale_numeric", True))
        encode_categoricals = bool(self.stage.params.get("encode_categoricals", True))
        test_size = float(self.stage.params.get("test_size", 0.2))
        task_type = str(self.stage.params.get("task_type", "classification")).strip().lower()
        require_binary_target = bool(
            self.stage.params.get("require_binary_target", task_type in {"classification", "anomaly"})
        )

        if not (0.0 < test_size < 1.0):
            raise ValueError(f"TabularPreprocessHandler: test_size must be in (0, 1), got {test_size}")

        # Drop rows where target is missing
        valid_target_mask = ~y.isna()
        dropped_rows = int((~valid_target_mask).sum())
        if dropped_rows > 0:
            X = X.loc[valid_target_mask]
            y = y.loc[valid_target_mask]

        if y.empty:
            raise ValueError("TabularPreprocessHandler: target column has no valid rows after filtering")

        class_counts = y.value_counts(dropna=False)
        if require_binary_target:
            if len(class_counts) != 2:
                raise ValueError(
                    "TabularPreprocessHandler: binary classification requires exactly 2 target classes. "
                    f"Found {len(class_counts)} classes: {list(class_counts.index)}"
                )
            if int(class_counts.min()) < 2:
                raise ValueError(
                    "TabularPreprocessHandler: each class needs at least 2 rows for stratified train/test split. "
                    f"Class counts: {class_counts.to_dict()}"
                )

        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

        # Numeric pipeline
        numeric_transformers = []
        if impute_missing:
            numeric_transformers.append(("imputer", SimpleImputer(strategy="median")))
        if scale_numeric:
            numeric_transformers.append(("scaler", StandardScaler()))

        if numeric_transformers:
            numeric_transformer = Pipeline(steps=numeric_transformers)
        else:
            numeric_transformer = "passthrough"

        # Categorical pipeline
        if encode_categoricals:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        else:
            # Current models do not support raw strings directly.
            categorical_transformer = "drop"

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        if task_type == "regression":
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y,
            )

        context["preprocessor"] = preprocessor
        context["X_train"] = X_train
        context["X_test"] = X_test
        context["y_train"] = y_train.to_numpy()
        context["y_test"] = y_test.to_numpy()
        context["feature_columns"] = X.columns.tolist()
        context["feature_dtypes"] = {col: str(X[col].dtype) for col in X.columns}
        context["target_class_counts"] = class_counts.to_dict()
        if require_binary_target:
            context["class_labels"] = class_counts.index.tolist()

        print(
            f"[tabular_preprocess] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, "
            f"y_train: {y_train.shape}, y_test: {y_test.shape}"
        )
        if dropped_rows > 0:
            print(f"[tabular_preprocess] Dropped {dropped_rows} rows with missing target values")

        return context
