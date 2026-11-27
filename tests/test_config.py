from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_config


def test_experiment_configs_extend_default():
    default = load_config()
    for path in sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml")):
        cfg = load_config(path)
        assert "extends" not in cfg
        assert cfg["data"] == default["data"], f"{path.name} must not change the data"
        assert cfg["model"]["name"] == path.stem


def test_extends_deep_merges(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: 3\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 5}\n")
    assert load_config(tmp_path / "child.yaml") == {"a": {"x": 1, "y": 5}, "b": 3}


def test_env_overrides_apply_to_every_config(tmp_path: Path, monkeypatch):
    (tmp_path / "over.yaml").write_text("tracking: {experiment_name: from-env}\n")
    monkeypatch.setenv("PII_CONFIG_OVERRIDES", str(tmp_path / "over.yaml"))
    cfg = load_config(PROJECT_ROOT / "configs" / "experiments" / "e1_small_cnn_subset.yaml")
    assert cfg["tracking"]["experiment_name"] == "from-env"
    assert cfg["model"]["name"] == "e1_small_cnn_subset"  # the rest is untouched
    # Explicit overrides still win over the environment file.
    assert (
        load_config(overrides={"tracking": {"experiment_name": "x"}})["tracking"]["experiment_name"]
        == "x"
    )


def test_docker_overrides_only_redirect_outputs():
    """The Docker override file must not change data, model or training settings."""
    from src.config import _load_yaml

    over = _load_yaml(PROJECT_ROOT / "configs" / "docker_overrides.yaml")
    assert set(over) == {"tracking", "artifacts", "validation", "evaluation", "train"}
    assert set(over["train"]) == {"report_dir"}
    assert set(over["validation"]) == {"report_dir"}


def test_extends_cycle_is_rejected(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")
