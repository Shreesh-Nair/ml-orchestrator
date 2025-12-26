from pathlib import Path

import pytest

from core.yaml_parser import load_yaml, parse_pipeline, PipelineConfig, Stage

def test_parse_valid_titanic_yaml() -> None:
    config = parse_pipeline("examples/titanic.yml")

    assert isinstance(config, PipelineConfig)
    assert config.pipeline_name == "titanic_survival"
    assert len(config.stages) == 3

    s0, s1, s2 = config.stages

    assert s0.name == "load_data"
    assert s0.type == "csv_loader"
    assert "source" in s0.params
    assert s0.params["source"] == "data/titanic.csv"

    assert s1.name == "preprocess"
    assert s1.type == "tabular_preprocess"
    assert s1.params.get("impute_missing") is True
    assert s1.params.get("scale_numeric") is True

    assert s2.name == "train"
    assert s2.type == "classification"
    # titanic.yml encodes models under 'models' at the stage level
    # you can assert it exists once you decide how to store it.
def test_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        parse_pipeline("examples/does_not_exist.yml")

def test_missing_pipeline_name_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_name.yml"
    p.write_text("stages: []", encoding="utf-8")

    with pytest.raises(ValueError, match="pipeline_name"):
        parse_pipeline(p)

def test_missing_stages_raises(tmp_path: Path) -> None:
    p = tmp_path / "no_stages.yml"
    p.write_text("pipeline_name: test_pipeline", encoding="utf-8")

    with pytest.raises(ValueError, match="stages"):
        parse_pipeline(p)

def test_stage_missing_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad_stage.yml"
    p.write_text(
        "pipeline_name: test\n"
        "stages:\n"
        "  - name: only_name\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing 'type'"):
        parse_pipeline(p)
