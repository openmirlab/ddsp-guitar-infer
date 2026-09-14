"""Offline checkpoint-manifest and explicit-device contract tests."""
from __future__ import annotations

import hashlib

import torch
import pytest

from ddsp_guitar_infer.api import _resolve_device
from ddsp_guitar_infer.utils import checkpoints


def test_manifest_drives_hub_resolution(monkeypatch, tmp_path):
    seen = {}
    real_config = checkpoints.checkpoint_config()
    fake_checkpoint = tmp_path / "x.ckpt"
    fake_checkpoint.write_bytes(b"pretend-checkpoint-bytes")
    fake_config = dict(real_config, sha256=hashlib.sha256(fake_checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: seen.update(kwargs) or str(fake_checkpoint))
    monkeypatch.setattr(checkpoints, "checkpoint_config", lambda: fake_config)
    assert checkpoints.resolve_checkpoint(cache_dir=str(tmp_path)) == fake_checkpoint
    assert seen["repo_id"] == real_config["repo"]
    assert seen["filename"] == real_config["filename"]
    assert seen["revision"] == real_config["revision"]


def test_checkpoint_config_has_pinned_sha256_and_revision():
    config = checkpoints.checkpoint_config()
    assert len(config["sha256"]) == 64
    bytes.fromhex(config["sha256"])  # raises ValueError if not valid lowercase hex
    assert config["sha256"] == config["sha256"].lower()
    assert isinstance(config["size_bytes"], int) and config["size_bytes"] > 0
    assert len(config["revision"]) == 40  # a full git commit sha, not the "main" branch ref
    bytes.fromhex(config["revision"])


def test_checkpoint_config_license_is_a_verified_assertion_not_a_bare_unknown():
    """`license` must be a checked SPDX id or the explicit NOASSERTION marker --

    never the bare, undocumented "unknown" this record used to carry (org
    constitution article 4/8: never claim a stronger license than upstream
    states, and record a verified NOASSERTION rather than silence).
    """
    config = checkpoints.checkpoint_config()
    assert config["license"] in {"NOASSERTION"} or "-" in config["license"]  # e.g. "Apache-2.0"
    assert config["license"] != "unknown"


def test_verification_raises_on_sha256_mismatch(monkeypatch, tmp_path):
    corrupted = tmp_path / "unified.ckpt"
    corrupted.write_bytes(b"not the real checkpoint bytes")
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: str(corrupted))
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_config",
        lambda: {"repo": "x/y", "filename": "unified.ckpt", "revision": "abc", "sha256": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="Integrity check FAILED"):
        checkpoints.resolve_checkpoint(cache_dir=str(tmp_path))


def test_verification_passes_on_sha256_match(monkeypatch, tmp_path):
    good = tmp_path / "unified.ckpt"
    good.write_bytes(b"the real checkpoint bytes")
    expected = hashlib.sha256(good.read_bytes()).hexdigest()
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: str(good))
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_config",
        lambda: {"repo": "x/y", "filename": "unified.ckpt", "revision": "abc", "sha256": expected},
    )
    assert checkpoints.resolve_checkpoint(cache_dir=str(tmp_path)) == good


def test_allow_download_false_skips_verification(monkeypatch, tmp_path):
    """cache_info()'s read-only path must never hash a (possibly huge) file."""
    mismatched = tmp_path / "unified.ckpt"
    mismatched.write_bytes(b"whatever is already cached locally")
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: str(mismatched))
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_config",
        lambda: {"repo": "x/y", "filename": "unified.ckpt", "revision": "abc", "sha256": "0" * 64},
    )
    assert checkpoints.resolve_checkpoint(cache_dir=str(tmp_path), allow_download=False) == mismatched


def test_explicit_path_override_skips_verification(monkeypatch, tmp_path):
    """An explicit path/env override is assumed intentional and never hashed."""
    custom = tmp_path / "my_checkpoint.ckpt"
    custom.write_bytes(b"whatever the caller says it is")
    monkeypatch.setattr(
        checkpoints,
        "checkpoint_config",
        lambda: {"repo": "x/y", "filename": "unified.ckpt", "revision": "abc", "sha256": "0" * 64},
    )
    assert checkpoints.resolve_checkpoint(str(custom)) == custom


def test_device_auto_and_explicit_contract(monkeypatch):
    monkeypatch.setattr("ddsp_guitar_infer.api.torch.cuda.is_available", lambda: False)
    assert _resolve_device(None) == torch.device("cpu")
    assert _resolve_device("auto") == torch.device("cpu")
    assert _resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError): _resolve_device("cuda:1")
    with pytest.raises(ValueError): _resolve_device("mps")
    monkeypatch.setattr("ddsp_guitar_infer.api.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("ddsp_guitar_infer.api.torch.cuda.device_count", lambda: 2)
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
