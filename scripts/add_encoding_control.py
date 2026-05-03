# Add encoding strategy control to gui/main.py after class imbalance section

with open('gui/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and insert the encoding control
old_text = '''        # Class imbalance controls (Phase 5)
        imbalance_row = QHBoxLayout()
        self.checkbox_class_weight = QCheckBox("Handle class imbalance (auto-weight minority class)")
        self.checkbox_class_weight.setChecked(False)
        self.checkbox_class_weight.setToolTip("Enable class weighting to penalize minority class misclassification (useful for imbalanced datasets).")
        imbalance_row.addWidget(self.checkbox_class_weight)
        imbalance_row.addStretch()
        top_layout.addLayout(imbalance_row)

        target_row = QHBoxLayout()'''

new_text = '''        # Class imbalance controls (Phase 5)
        imbalance_row = QHBoxLayout()
        self.checkbox_class_weight = QCheckBox("Handle class imbalance (auto-weight minority class)")
        self.checkbox_class_weight.setChecked(False)
        self.checkbox_class_weight.setToolTip("Enable class weighting to penalize minority class misclassification (useful for imbalanced datasets).")
        imbalance_row.addWidget(self.checkbox_class_weight)
        imbalance_row.addStretch()
        top_layout.addLayout(imbalance_row)

        # Encoding strategy control (Phase 5)
        encoding_row = QHBoxLayout()
        encoding_row.addWidget(QLabel("Encoding strategy:"))
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItem("One-Hot (standard, creates many columns)", "onehot")
        self.encoding_combo.addItem("Ordinal (orderable categories, fewer columns)", "ordinal")
        self.encoding_combo.addItem("Target (mean-target-encoded, uses target info)", "target")
        self.encoding_combo.setCurrentIndex(0)
        self.encoding_combo.setToolTip("One-Hot: standard and robust. Ordinal: for orderable categories. Target: uses target mean (may overfit on small data).")
        encoding_row.addWidget(self.encoding_combo)
        encoding_row.addStretch()
        top_layout.addLayout(encoding_row)

        target_row = QHBoxLayout()'''

content = content.replace(old_text, new_text)

with open('gui/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Added encoding_combo control to gui/main.py")
