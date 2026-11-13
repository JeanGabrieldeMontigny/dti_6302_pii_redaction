"""Configuration loading. Relative paths in configs resolve against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config, optionally deep-merging ``overrides`` on top."""
    with open(resolve_path(path) if path else DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


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
