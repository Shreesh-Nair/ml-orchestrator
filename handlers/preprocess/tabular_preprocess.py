from __future__ import annotations

from typing import Any, Dict

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

        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()

        if X.shape[1] == 0:
            raise ValueError("TabularPreprocessHandler: dataset must include at least one feature column")
        if len(df) < 4:
            raise ValueError("TabularPreprocessHandler: at least 4 rows are required for train/test split")

        # Read preprocessing params
        impute_missing = bool(self.stage.params.get("impute_missing", True))
        scale_numeric = bool(self.stage.params.get("scale_numeric", True))
        encode_categoricals = bool(self.stage.params.get("encode_categoricals", True))
        test_size = float(self.stage.params.get("test_size", 0.2))
        random_state = int(self.stage.params.get("random_state", context.get("_random_seed", 42)))
        task_type = str(self.stage.params.get("task_type", "classification")).strip().lower()
        require_binary_target = bool(
            self.stage.params.get("require_binary_target", task_type in {"classification", "anomaly"})
        )
        rare_category_min_freq = float(self.stage.params.get("rare_category_min_freq", 0.0))
        text_extract = bool(self.stage.params.get("text_extract", False))
        text_drop_original = bool(self.stage.params.get("text_drop_original", False))
        text_feature_columns = self.stage.params.get("text_feature_columns")

        # Optional: extract date/time features from parseable datetime columns
        date_extract = bool(self.stage.params.get("date_extract", False))
        if date_extract:
            from pandas.api import types as ptypes

            for col in list(X.columns):
                # only attempt to parse string-like or datetime dtypes
                is_str = ptypes.is_string_dtype(X[col]) or ptypes.is_object_dtype(X[col])
                is_dt = ptypes.is_datetime64_any_dtype(X[col])
                if not (is_str or is_dt):
                    continue

                try:
                    parsed = pd.to_datetime(X[col], errors="coerce")
                except Exception:
                    parsed = pd.Series(index=X.index, dtype="datetime64[ns]")

                non_na = int(parsed.notna().sum())
                # If a reasonable fraction parses as datetime, create extraction columns
                if non_na >= max(1, int(0.5 * len(parsed))):
                    X[f"{col}__year"] = parsed.dt.year
                    X[f"{col}__month"] = parsed.dt.month
                    X[f"{col}__day"] = parsed.dt.day
                    X[f"{col}__hour"] = parsed.dt.hour
                    X[f"{col}__weekday"] = parsed.dt.weekday
                    # remove original text/date column so downstream categorical
                    # pipelines do not attempt to impute/encode raw strings
                    X.drop(columns=[col], inplace=True)

        # Optional: extract basic text features from selected or inferred text columns.
        if text_extract:
            from pandas.api import types as ptypes

            if isinstance(text_feature_columns, list):
                candidate_text_cols = [
                    c for c in text_feature_columns if c in X.columns and (ptypes.is_string_dtype(X[c]) or ptypes.is_object_dtype(X[c]))
                ]
            else:
                candidate_text_cols = [
                    c for c in X.columns if (ptypes.is_string_dtype(X[c]) or ptypes.is_object_dtype(X[c]))
                ]

            for col in candidate_text_cols:
                sample = X[col].dropna().astype(str).head(30)
                looks_date_like = bool(
                    len(sample) > 0
                    and sample.str.contains(r"\d", regex=True).mean() >= 0.6
                    and sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.6
                )
                if looks_date_like:
                    continue

                text_series = X[col].fillna("").astype(str)
                X[f"{col}__char_len"] = text_series.str.len()
                X[f"{col}__word_count"] = text_series.str.split().str.len()

                if text_drop_original:
                    X.drop(columns=[col], inplace=True)

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

        # Determine categorical and numeric columns, excluding date-like object columns
        from pandas.api import types as ptypes
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype.name in {"object", "category", "bool"}:
                sample = X[col].dropna().astype(str).head(30)
                looks_date_like = bool(
                    len(sample) > 0
                    and sample.str.contains(r"\d", regex=True).mean() >= 0.6
                    and sample.str.contains(r"[-/:T]", regex=True).mean() >= 0.6
                )

                if looks_date_like:
                    try:
                        parsed = pd.to_datetime(X[col], errors="coerce")
                    except Exception:
                        parsed = pd.Series(index=X.index, dtype="datetime64[ns]")
                else:
                    parsed = pd.Series(index=X.index, dtype="datetime64[ns]")

                if int(parsed.notna().sum()) >= max(1, int(0.5 * len(parsed))):
                    # treat as date-like -> will be handled by extraction or numeric pipeline
                    continue
                categorical_cols.append(col)
        numeric_cols = [c for c in X.columns if ptypes.is_numeric_dtype(X[c]) or ptypes.is_datetime64_any_dtype(X[c])]

        # Optional: group rare categories to a shared bucket before one-hot encoding.
        if rare_category_min_freq > 0 and categorical_cols:
            for col in categorical_cols:
                value_freq = X[col].value_counts(normalize=True, dropna=True)
                rare_values = value_freq[value_freq < rare_category_min_freq].index
                if len(rare_values) > 0:
                    X[col] = X[col].replace(rare_values, "__OTHER__")

        # Normalize missing categorical values so most_frequent imputer sees consistent null markers.
        if categorical_cols:
            X[categorical_cols] = X[categorical_cols].where(X[categorical_cols].notna(), np.nan)

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
                X_processed, y, test_size=test_size, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed,
                y,
                test_size=test_size,
                random_state=random_state,
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
        context["_random_seed"] = random_state
        if require_binary_target:
            context["class_labels"] = class_counts.index.tolist()

        print(
            f"[tabular_preprocess] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, "
            f"y_train: {y_train.shape}, y_test: {y_test.shape}"
        )
        if dropped_rows > 0:
            print(f"[tabular_preprocess] Dropped {dropped_rows} rows with missing target values")

        return context
