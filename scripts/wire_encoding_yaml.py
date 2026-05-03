# Wire encoding_strategy to YAML generation in gui/main.py

with open('gui/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and add encoding_strategy parameter to preprocess params
old_yaml_preprocess = '''        validation_strategy = "stratified"  # Default
        if hasattr(self, "validation_strategy_combo"):
            validation_strategy = self.validation_strategy_combo.currentData() or "stratified"

        stages.append(
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "target_column": self.target_column,
                    "task_type": self.selected_task,
                    "require_binary_target": bool(task_cfg.get("require_binary_target", False)),
                    "scale_numeric": self.scale_checkbox.isChecked(),
                    "encode_categoricals": self.encode_checkbox.isChecked(),
                    "test_size": float(self.test_size_spin.value()),
                    "random_state": int(self.random_seed_spin.value()),
                    "validation_strategy": validation_strategy,
                },
            }
        )'''

new_yaml_preprocess = '''        validation_strategy = "stratified"  # Default
        if hasattr(self, "validation_strategy_combo"):
            validation_strategy = self.validation_strategy_combo.currentData() or "stratified"

        # Get encoding strategy setting (Phase 5)
        encoding_strategy = "onehot"
        if hasattr(self, "encoding_combo"):
            encoding_strategy = self.encoding_combo.currentData() or "onehot"

        stages.append(
            {
                "name": "preprocess",
                "type": "tabular_preprocess",
                "params": {
                    "target_column": self.target_column,
                    "task_type": self.selected_task,
                    "require_binary_target": bool(task_cfg.get("require_binary_target", False)),
                    "scale_numeric": self.scale_checkbox.isChecked(),
                    "encode_categoricals": self.encode_checkbox.isChecked(),
                    "test_size": float(self.test_size_spin.value()),
                    "random_state": int(self.random_seed_spin.value()),
                    "validation_strategy": validation_strategy,
                    "encoding_strategy": encoding_strategy,
                },
            }
        )'''

content = content.replace(old_yaml_preprocess, new_yaml_preprocess)

with open('gui/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Wired encoding_strategy to YAML generation in gui/main.py")
