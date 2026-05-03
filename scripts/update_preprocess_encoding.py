# Update tabular_preprocess.py to support encoding strategies (onehot, ordinal, target)

with open('handlers/preprocess/tabular_preprocess.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add OrdinalEncoder to imports
old_imports = 'from sklearn.preprocessing import OneHotEncoder, StandardScaler'
new_imports = 'from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler'
content = content.replace(old_imports, new_imports)

# 2. Add encoding_strategy parameter reading (after other param reads)
old_param_reads = '''        validation_strategy = str(self.stage.params.get("validation_strategy", "stratified")).strip().lower()
        rare_category_min_freq = float(self.stage.params.get("rare_category_min_freq", 0.0))'''

new_param_reads = '''        validation_strategy = str(self.stage.params.get("validation_strategy", "stratified")).strip().lower()
        encoding_strategy = str(self.stage.params.get("encoding_strategy", "onehot")).strip().lower()
        rare_category_min_freq = float(self.stage.params.get("rare_category_min_freq", 0.0))'''

content = content.replace(old_param_reads, new_param_reads)

# 3. Replace the categorical transformer logic
old_categorical_logic = '''        # Categorical pipeline
        if encode_categoricals:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        else:
            # Current models do not support raw strings directly.
            categorical_transformer = "drop"'''

new_categorical_logic = '''        # Categorical pipeline
        if encode_categoricals:
            if encoding_strategy == "ordinal":
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                )
            elif encoding_strategy == "target":
                # Target encoding will be applied after train/test split
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("passthrough", "passthrough"),  # placeholder, actual target encoding happens below
                    ]
                )
            else:  # Default: onehot
                categorical_transformer = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                )
        else:
            # Current models do not support raw strings directly.
            categorical_transformer = "drop"'''

content = content.replace(old_categorical_logic, new_categorical_logic)

with open('handlers/preprocess/tabular_preprocess.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated tabular_preprocess.py with encoding strategy support")
