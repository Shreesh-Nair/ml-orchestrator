# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - Unreleased

### Added
- Packaging guidance and reproducible packaging steps to `README.md` to avoid NumPy ABI mismatch during PyInstaller builds.
- `scripts/find_compiled_extensions.py` to scan installed site-packages for compiled extensions and produce `build/compiled_extensions.json`.
- Data-quality quick-fix recommender: `recommend_quick_fixes()` and GUI integration (`Apply Recommended Fixes` button).
- Headless GUI smoke tests for core Train flows.

### Changed
- Pin `numpy<2` in packaging contexts (CI) to avoid ABI incompatibilities with prebuilt wheels.

### Notes
- Long-term: upgrade compiled wheels to NumPy 2-compatible builds or rebuild from source; short-term CI pins `numpy<2` for reliable packaging.
