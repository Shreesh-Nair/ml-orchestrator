# handlers/preprocess/tabular_preprocess.py
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
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
      - One-hot encodes categoricals
      - Scales numerical features
      - Train/test split
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

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Basic heuristics: object/category → categorical, rest → numeric
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

        impute_missing = bool(self.stage.params.get("impute_missing", True))
        scale_numeric = bool(self.stage.params.get("scale_numeric", True))

        numeric_transformers = []
        if impute_missing:
            numeric_transformers.append(("imputer", SimpleImputer(strategy="median")))
        if scale_numeric:
            numeric_transformers.append(("scaler", StandardScaler()))

        # If no transforms, use passthrough
        if numeric_transformers:
            numeric_transformer = Pipeline(steps=numeric_transformers)
        else:
            numeric_transformer = "passthrough"

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        context["preprocessor"] = preprocessor
        context["X_train"] = X_train
        context["X_test"] = X_test
        context["y_train"] = y_train.to_numpy()
        context["y_test"] = y_test.to_numpy()

        print(
            f"[tabular_preprocess] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, "
            f"y_train: {y_train.shape}, y_test: {y_test.shape}"
        )

        return context
