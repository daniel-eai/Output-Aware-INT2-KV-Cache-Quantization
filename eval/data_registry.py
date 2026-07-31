#!/usr/bin/env python3
"""Pinned benchmark data downloads with checksum verification."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "data_manifest.json"


class DataIntegrityError(RuntimeError):
    """Raised when downloaded benchmark data does not match the manifest."""


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise DataIntegrityError("unsupported data manifest schema")
    return manifest


def dataset_entry(name: str) -> dict[str, Any]:
    datasets = load_manifest()["datasets"]
    try:
        return dict(datasets[name])
    except KeyError as exc:
        raise KeyError(f"unknown pinned dataset: {name}") from exc


def data_cache_dir() -> Path:
    override = os.environ.get("OPTR_DATA_CACHE")
    root = Path(override).expanduser() if override else Path.home() / ".cache" / "optr"
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> Path:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise DataIntegrityError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return path


def _download_https(url: str, destination: Path, expected_sha256: str) -> Path:
    if destination.exists():
        return verify_file(destination, expected_sha256)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, output)
        verify_file(temporary, expected_sha256)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _materialize_huggingface(entry: dict[str, Any]) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            repo_type=entry["repo_type"],
            revision=entry["revision"],
            cache_dir=data_cache_dir() / "huggingface",
        )
    )
    return verify_file(path, entry["sha256"])


def _materialize_gzip(entry: dict[str, Any]) -> Path:
    compressed = _download_https(
        entry["url"], data_cache_dir() / entry["cache_name"], entry["sha256"]
    )
    content = data_cache_dir() / entry["content_name"]
    if not content.exists():
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{content.name}.", dir=content.parent
        )
        temporary = Path(temporary_name)
        try:
            with gzip.open(compressed, "rb") as source, os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(source, output)
            verify_file(temporary, entry["content_sha256"])
            os.replace(temporary, content)
        finally:
            temporary.unlink(missing_ok=True)

    verify_file(content, entry["content_sha256"])
    actual_md5 = file_md5(content)
    if actual_md5 != entry["content_md5"]:
        raise DataIntegrityError(
            f"content MD5 mismatch for {content}: "
            f"expected {entry['content_md5']}, got {actual_md5}"
        )
    return content


def materialize_dataset(name: str) -> Path:
    entry = dataset_entry(name)
    kind = entry["kind"]
    if kind == "huggingface":
        path = _materialize_huggingface(entry)
    elif kind == "https":
        path = _download_https(
            entry["url"], data_cache_dir() / entry["cache_name"], entry["sha256"]
        )
    elif kind == "https_gzip":
        path = _materialize_gzip(entry)
    else:
        raise DataIntegrityError(f"unsupported dataset source kind: {kind}")
    return path


def validate_manifest() -> None:
    manifest = load_manifest()
    required = {"aime24", "aime25", "gpqa_diamond", "lcb_v6", "mbpp_plus"}
    datasets = manifest["datasets"]
    if set(datasets) != required:
        raise DataIntegrityError(
            f"manifest datasets differ from the supported set: {sorted(datasets)}"
        )
    for name, entry in datasets.items():
        sha = entry.get("sha256", "")
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise DataIntegrityError(f"invalid SHA256 for {name}")
        if entry["kind"] == "huggingface" and len(entry.get("revision", "")) != 40:
            raise DataIntegrityError(f"invalid immutable revision for {name}")
        if int(entry.get("expected_examples", 0)) <= 0:
            raise DataIntegrityError(f"invalid expected example count for {name}")


if __name__ == "__main__":
    validate_manifest()
    print(f"validated {MANIFEST_PATH}")
