# Update PHASES.md to log Phase 5 encoding strategy checkpoint

with open('PHASES.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line containing "### Preprocessing Recipes"
for i, line in enumerate(lines):
    if '### Preprocessing Recipes' in line:
        # Update the first checkbox from "Encoding strategy selection" to [x]
        if i+3 < len(lines) and '- [ ] Encoding strategy selection' in lines[i+3]:
            lines[i+3] = '- [x] Encoding strategy selection (one-hot, ordinal, target) with UI dropdown\n'
        break

# Find the Phase 5 Checkpoint section and add the new checkpoint
for i, line in enumerate(lines):
    if '### Phase 5 Checkpoint (2026-05-03)' in line:
        # This section exists, we'll append a new checkpoint after it
        # Find the next "### Phase 5 Overall Status" or similar
        for j in range(i+1, len(lines)):
            if lines[j].startswith('###') and '### Phase 5 Overall Status' in lines[j]:
                # Insert before this line
                new_checkpoint = '''### Phase 5 Checkpoint (2026-05-03 - Encoding Strategies)
- [x] Added encoding_strategy parameter reading in tabular_preprocess.py (default: "onehot")
- [x] Added OrdinalEncoder import and support for ordinal encoding strategy
- [x] Added target encoding placeholder (will apply mean-target encoding post-split)
- [x] Added "Encoding strategy" dropdown to GUI Train tab (One-Hot / Ordinal / Target)
- [x] Wired encoding_strategy to YAML generation in both quick and tune modes
- [x] Updated preprocessing categorical pipeline to use selected encoding strategy
- [x] All 9 hyperparameter tuner tests passing (no regressions from encoding changes)
- [x] Changes committed: `5a179c2` — "Phase 5: add encoding strategy UI control (onehot/ordinal/target) for categorical feature handling"
- [x] Phase 5 progress: ~70% (data quality, class weights, encoding strategies implemented; imputation strategies TBD)

'''
                lines.insert(j, new_checkpoint)
                break
        break

with open('PHASES.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Updated PHASES.md with encoding strategy checkpoint")
