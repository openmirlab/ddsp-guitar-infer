"""Offline checkpoint-manifest and explicit-device contract tests."""
from __future__ import annotations

import torch
import pytest

from ddsp_guitar_infer.api import _resolve_device
from ddsp_guitar_infer.utils import checkpoints


def test_manifest_drives_hub_resolution(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(checkpoints, "hf_hub_download", lambda **kwargs: seen.update(kwargs) or str(tmp_path / "x.ckpt"))
    assert checkpoints.resolve_checkpoint(cache_dir=str(tmp_path)) == tmp_path / "x.ckpt"
    config = checkpoints.checkpoint_config()
    assert seen["repo_id"] == config["repo"]
    assert seen["filename"] == config["filename"]
    assert config["integrity"] == "unavailable"


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
