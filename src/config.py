"""Configuration loading. Relative paths in configs resolve against the repository root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Optional YAML file deep-merged over *every* loaded config. The Docker Airflow service uses it to
# send MLflow runs, models and reports to its own volume, whichever experiment config is chosen.
OVERRIDES_ENV = "PII_CONFIG_OVERRIDES"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config, then apply ``$PII_CONFIG_OVERRIDES`` and ``overrides`` (in that order).

    A config may start with ``extends: <path>`` (relative to its own file) to inherit from another
    config; its own keys are deep-merged over the parent's. Experiment configs use this to state
    only what they change.
    """
    cfg = _load_yaml(resolve_path(path) if path else DEFAULT_CONFIG)
    if env_file := os.environ.get(OVERRIDES_ENV):
        _deep_merge(cfg, _load_yaml(resolve_path(env_file)))
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _load_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"config extends cycle: {' -> '.join(map(str, (*_seen, path)))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    base = _load_yaml(path.parent / parent, (*_seen, path))
    _deep_merge(base, cfg)
    return base


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_tracking_uri(uri: str) -> str:
    """Make ``sqlite:///relative.db`` URIs absolute so runs land in one place regardless of cwd."""
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not uri.startswith(prefix + "/"):
        return prefix + str(resolve_path(uri[len(prefix) :]))
    return uri


def _deep_merge(base: dict, other: dict) -> None:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
