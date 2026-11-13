"""Download the raw Nemotron-PII parquet files at a pinned revision and record a manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_filename(split: str) -> str:
    return f"{split}.parquet"


def write_manifest(raw_dir: Path, repo: str, revision: str, splits: list[str]) -> dict:
    """Record checksums for raw files placed by hand (tests, offline copies)."""
    files = {raw_filename(s): sha256_file(raw_dir / raw_filename(s)) for s in splits}
    manifest = {"repo": repo, "revision": revision, "files": files}
    (raw_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(raw_dir: Path, splits: list[str]) -> dict:
    """Fail if the raw files on disk are not the ones the manifest describes."""
    path = raw_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the ingest step first")
    manifest = json.loads(path.read_text())
    for split in splits:
        name = raw_filename(split)
        expected = manifest["files"].get(name)
        actual = sha256_file(raw_dir / name)
        if actual != expected:
            raise RuntimeError(
                f"{raw_dir / name} does not match {path} (sha256 {actual[:12]} != "
                f"{str(expected)[:12]}); re-run ingest"
            )
    return manifest


def download_raw(repo: str, revision: str, splits: list[str], raw_dir: Path) -> dict:
    """Fetch each split's parquet file into ``raw_dir`` (skipped if the manifest already matches)."""
    from huggingface_hub import hf_hub_download

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("revision") == revision and all(
            (raw_dir / raw_filename(s)).exists()
            and sha256_file(raw_dir / raw_filename(s)) == manifest["files"].get(raw_filename(s))
            for s in splits
        ):
            log.info("Raw data already present at revision %s; skipping download", revision)
            return manifest

    files = {}
    for split in splits:
        cached = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=f"data/{split}-00000-of-00001.parquet",
            revision=revision,
        )
        target = raw_dir / raw_filename(split)
        target.write_bytes(Path(cached).read_bytes())
        files[target.name] = sha256_file(target)
        log.info("Downloaded %s (%s)", target, files[target.name][:12])

    manifest = {"repo": repo, "revision": revision, "files": files}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
