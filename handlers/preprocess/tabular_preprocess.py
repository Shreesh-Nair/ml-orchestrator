# handlers/preprocess/tabular_preprocess.py
from __future__ import annotations

from typing import Dict, Any

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

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Basic heuristics: object/category → categorical, rest → numeric
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

        # Read preprocessing params (defaults to True/0.2 if missing)
        impute_missing = bool(self.stage.params.get("impute_missing", True))
        scale_numeric = bool(self.stage.params.get("scale_numeric", True))
        encode_categoricals = bool(self.stage.params.get("encode_categoricals", True))  # ✅ FIXED
        test_size = float(self.stage.params.get("test_size", 0.2))  # ✅ FIXED

        # --- 1. Numeric Pipeline ---
        numeric_transformers = []
        if impute_missing:
            numeric_transformers.append(("imputer", SimpleImputer(strategy="median")))
        if scale_numeric:
            numeric_transformers.append(("scaler", StandardScaler()))

        if numeric_transformers:
            numeric_transformer = Pipeline(steps=numeric_transformers)
        else:
            numeric_transformer = "passthrough"

        # --- 2. Categorical Pipeline ---
        if encode_categoricals:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        else:
            # If user unchecks "Encode", we drop categoricals or pass raw?
            # Standard sklearn models (RF/LogReg) crash on raw strings.
            # Best behavior here: if encoding disabled, we just drop categorical columns
            # OR pass them through if using a model that handles them (like CatBoost).
            # For your current RandomForest, dropping them is safer to avoid crashes.
            categorical_transformer = "drop"

        # --- 3. Combine ---
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        # --- 4. Split (using GUI test_size) ---
        task_type = self.stage.params.get("task_type", "classification")
        if task_type == "regression":
            # Regression: no stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42  # ✅ FIXED
            )
        else:
            # Classification: use stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42, stratify=y  # ✅ FIXED
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
