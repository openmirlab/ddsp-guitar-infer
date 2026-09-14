"""checkpoints.py — checkpoint resolution + integrity verification (★ load-bearing).

Resolves a local `.ckpt` path in priority order: explicit path argument,
the DDSP_GUITAR_WEIGHTS env var (for CI/offline use), then a Hugging Face
Hub download of erl-j/ddsp-guitar-unified pinned to a specific commit
revision. This is the only place network access happens in the package.

A hub-resolved artifact is sha256-verified against `config/checkpoints.toml`
whenever a real download is permitted (`allow_download=True`, the `load()`
path) -- matching adtof-infer's precedent: an explicitly supplied path or
`DDSP_GUITAR_WEIGHTS` override is assumed intentional (a test fixture or a
caller's own checkpoint) and is never hashed. `allow_download=False`
(`cache_info()`'s read-only status check) never verifies either, so status
reporting never pays a ~360MB hashing cost.

Reads: huggingface_hub.hf_hub_download
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

def checkpoint_config() -> dict:
    path = Path(__file__).parents[1] / "config" / "checkpoints.toml"
    with path.open("rb") as handle:
        data = tomllib.load(handle).get("default")
    if not isinstance(data, dict) or not data.get("repo") or not data.get("filename"):
        raise ValueError(f"invalid checkpoint config: {path}")
    return data

DEFAULT_REPO = checkpoint_config()["repo"]
DEFAULT_FILENAME = checkpoint_config()["filename"]

#: Environment variable that, when set, points at a local checkpoint file and
#: bypasses the Hugging Face download. Used by CI (no network access) and by
#: anyone who already has the weights on disk.
WEIGHTS_ENV_VAR = "DDSP_GUITAR_WEIGHTS"


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected_sha256: str) -> None:
    """Raise RuntimeError if `path`'s sha256 doesn't match `expected_sha256`.

    A blank `expected_sha256` (the documented article-4 "unavailable"
    exception shape) is a no-op -- there is nothing to check against.
    """
    if not expected_sha256:
        return
    actual = _sha256_of_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Integrity check FAILED for checkpoint {path}: "
            f"expected sha256={expected_sha256}, got sha256={actual}. "
            "The download may be corrupted, or the upstream file may have "
            "changed. Refusing to use it -- delete the cached file and retry, "
            f"or set {WEIGHTS_ENV_VAR}=/path/to/known-good.ckpt to bypass the "
            "hub download entirely."
        )


def resolve_checkpoint(
    path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = True,
) -> Path:
    """Return a local path to the checkpoint, downloading if needed.

    Resolution order: explicit ``path`` argument, then the
    ``DDSP_GUITAR_WEIGHTS`` environment variable, then a Hugging Face Hub
    download of :data:`DEFAULT_REPO`/:data:`DEFAULT_FILENAME` pinned to the
    configured ``revision``. Set ``allow_download=False`` to inspect the
    Hugging Face cache without contacting the network.

    A hub-resolved artifact is sha256-verified against the packaged
    checkpoint config, but only when ``allow_download=True`` -- see the
    module docstring for why the two override paths and the read-only
    ``allow_download=False`` path skip verification.
    """
    if path is None:
        path = os.environ.get(WEIGHTS_ENV_VAR)

    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    metadata = checkpoint_config()
    download_path = Path(
        hf_hub_download(
            repo_id=metadata["repo"], filename=metadata["filename"], revision=metadata["revision"],
            local_dir=cache_dir,
            local_files_only=not allow_download,
        )
    )
    if allow_download:
        _verify_checksum(download_path, metadata.get("sha256", ""))
    return download_path

__all__ = ["resolve_checkpoint", "DEFAULT_REPO", "DEFAULT_FILENAME", "WEIGHTS_ENV_VAR"]
