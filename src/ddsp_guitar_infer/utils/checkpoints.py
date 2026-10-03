"""checkpoints.py — checkpoint resolution + integrity verification (★ load-bearing).

Resolves a local `.ckpt` path in priority order: explicit path argument,
the DDSP_GUITAR_WEIGHTS env var (for CI/offline use), then a Hugging Face
Hub download of erl-j/ddsp-guitar-unified pinned to a specific commit
revision. Callers may supply a TOML manifest and metadata overrides; an
HTTP(S) URL override uses a digest-keyed cache. This is the only place
network access happens in the package.

A hub-resolved artifact is sha256-verified against `config/checkpoints.toml`
whenever a real download is permitted (`allow_download=True`, the `load()`
path) -- matching adtof-infer's precedent: an explicitly supplied path or
`DDSP_GUITAR_WEIGHTS` override is assumed intentional (a test fixture or a
caller's own checkpoint) and is never hashed. `allow_download=False`
(`cache_info()`'s read-only status check) never verifies either, so status
reporting never pays a ~360MB hashing cost.

Reads: huggingface_hub.hf_hub_download · urllib.request.urlopen
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

def checkpoint_config(
    config_path: Optional[str] = None,
    checkpoint_overrides: Optional[dict] = None,
) -> dict:
    """Read the package manifest or a caller's manifest, then apply metadata overrides."""
    path = (
        Path(config_path) if config_path is not None
        else Path(__file__).parents[1] / "config" / "checkpoints.toml"
    )
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle).get("default")
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid checkpoint config: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"invalid checkpoint config: {path}")
    metadata = dict(data)
    if checkpoint_overrides:
        metadata.update(checkpoint_overrides)
    filename = metadata.get("filename")
    digest = metadata.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"invalid checkpoint filename in config: {path}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"invalid checkpoint SHA-256 in config: {path}")
    if "url" in metadata:
        if not isinstance(metadata["url"], str) or not metadata["url"]:
            raise ValueError(f"invalid checkpoint URL in config: {path}")
    elif (
        not isinstance(metadata.get("repo"), str) or not metadata["repo"]
        or not isinstance(metadata.get("revision"), str) or not metadata["revision"]
    ):
        raise ValueError(f"invalid checkpoint config: {path}")
    return metadata

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


def _resolve_url_checkpoint(metadata: dict, cache_dir: Optional[str], allow_download: bool) -> Path:
    """Cache a caller-supplied URL by digest, verifying before atomic installation."""
    url = metadata["url"]
    if not isinstance(url, str):
        raise ValueError("checkpoint url must be an HTTP(S) URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("checkpoint url must be an HTTP(S) URL")
    expected = metadata.get("sha256", "")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("URL checkpoint requires a lowercase SHA-256 digest")
    filename = metadata["filename"]
    if not isinstance(filename, str) or filename in {"", ".", ".."} or Path(filename).name != filename:
        raise ValueError("URL checkpoint filename must be a plain filename")
    root = (
        Path(cache_dir).expanduser() if cache_dir is not None
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "ddsp-guitar-infer" / "checkpoints"
    )
    target = root / expected / filename
    if target.is_file():
        if allow_download:
            _verify_checksum(target, expected)
        return target
    if not allow_download:
        raise FileNotFoundError(f"Checkpoint not cached: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url) as source, tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".download-", delete=False
        ) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(source, output)
        _verify_checksum(temporary, expected)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def resolve_checkpoint(
    path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = True,
    config_path: Optional[str] = None,
    checkpoint_overrides: Optional[dict] = None,
) -> Path:
    """Return a local path to the checkpoint, downloading if needed.

    Resolution order: explicit ``path`` argument, then the
    ``DDSP_GUITAR_WEIGHTS`` environment variable, then a Hugging Face Hub
    download of :data:`DEFAULT_REPO`/:data:`DEFAULT_FILENAME` pinned to the
    configured ``revision``. ``config_path`` selects a caller manifest and
    ``checkpoint_overrides`` replaces individual metadata fields, including
    a direct HTTP(S) ``url`` and its required ``sha256``. Set
    ``allow_download=False`` to inspect the matching cache without network.

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

    metadata = (
        checkpoint_config() if config_path is None and checkpoint_overrides is None
        else checkpoint_config(config_path, checkpoint_overrides)
    )
    if "url" in metadata:
        return _resolve_url_checkpoint(metadata, cache_dir, allow_download)
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

__all__ = ["checkpoint_config", "resolve_checkpoint", "DEFAULT_REPO", "DEFAULT_FILENAME", "WEIGHTS_ENV_VAR"]
