"""Checkpoint management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "erl-j/ddsp-guitar-unified"
DEFAULT_FILENAME = "unified.ckpt"


def resolve_checkpoint(path: Optional[str] = None, cache_dir: Optional[str] = None) -> Path:
    """Return a local path to the checkpoint, downloading if needed."""
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    download_path = hf_hub_download(
        repo_id=DEFAULT_REPO,
        filename=DEFAULT_FILENAME,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    return Path(download_path)

__all__ = ["resolve_checkpoint", "DEFAULT_REPO", "DEFAULT_FILENAME"]
