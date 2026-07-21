"""checkpoints.py — checkpoint resolution (★ load-bearing).

Resolves a local `.ckpt` path in priority order: explicit path argument,
the DDSP_GUITAR_WEIGHTS env var (for CI/offline use), then a Hugging Face
Hub download of erl-j/ddsp-guitar-unified. This is the only place network
access happens in the package.

Reads: huggingface_hub.hf_hub_download
"""

from __future__ import annotations

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


def resolve_checkpoint(
    path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = True,
) -> Path:
    """Return a local path to the checkpoint, downloading if needed.

    Resolution order: explicit ``path`` argument, then the
    ``DDSP_GUITAR_WEIGHTS`` environment variable, then a Hugging Face Hub
    download of :data:`DEFAULT_REPO`/:data:`DEFAULT_FILENAME`. Set
    ``allow_download=False`` to inspect the Hugging Face cache without
    contacting the network.
    """
    if path is None:
        path = os.environ.get(WEIGHTS_ENV_VAR)

    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    metadata = checkpoint_config()
    download_path = hf_hub_download(
        repo_id=metadata["repo"], filename=metadata["filename"], revision=metadata["revision"],
        local_dir=cache_dir,
        local_files_only=not allow_download,
    )
    return Path(download_path)

__all__ = ["resolve_checkpoint", "DEFAULT_REPO", "DEFAULT_FILENAME", "WEIGHTS_ENV_VAR"]
