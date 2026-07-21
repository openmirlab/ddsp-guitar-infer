"""Offline lifecycle contract tests for GuitarSynthSession."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ddsp_guitar_infer import api
from ddsp_guitar_infer.api import GuitarSynthSession
from ddsp_guitar_infer.utils import checkpoints


class SynthDouble:
    def __init__(self, marker):
        self.marker = marker

    def render_midi(self, *args, **kwargs):
        return (self.marker, args, kwargs)


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match="not loaded"):
        GuitarSynthSession().infer("song.mid")


def test_load_is_idempotent_and_forwards_cuda_device(monkeypatch):
    calls = []

    def fake_load(checkpoint, device, cache_dir):
        calls.append((checkpoint, device, cache_dir))
        return SynthDouble(len(calls))

    monkeypatch.setattr("ddsp_guitar_infer.api.load_synth", fake_load)
    session = GuitarSynthSession("weights.ckpt", "cuda:1", "cache")
    assert session.load() is session
    assert session.load() is session
    assert calls == [("weights.ckpt", "cuda:1", "cache")]
    assert session.infer("song.mid") == (1, ("song.mid",), {})
    assert session.render_midi("song.mid") == (1, ("song.mid",), {})


def test_release_then_load_reconstructs(monkeypatch):
    created = []
    monkeypatch.setattr(
        "ddsp_guitar_infer.api.load_synth",
        lambda *args: created.append(SynthDouble(len(created) + 1)) or created[-1],
    )
    session = GuitarSynthSession().load()
    session.release()
    assert session.status == "released"
    session.load()
    assert len(created) == 2


def test_close_is_terminal_and_idempotent(monkeypatch):
    monkeypatch.setattr("ddsp_guitar_infer.api.load_synth", lambda *args: SynthDouble(1))
    session = GuitarSynthSession().load()
    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="closed"):
        session.render_midi("song.mid")


def test_context_manager_loads_and_closes(monkeypatch):
    monkeypatch.setattr("ddsp_guitar_infer.api.load_synth", lambda *args: SynthDouble(1))
    session = GuitarSynthSession()
    with session as ready:
        assert ready is session
        assert session.status == "ready"
    assert session.status == "closed"


class ModelDouble:
    def __init__(self, config):
        self.config = config
        self.device = None

    def load_state_dict(self, state_dict, strict):
        assert strict is True

    def to(self, device):
        self.device = device

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def _checkpoint_payload():
    return {
        "hyper_parameters": {
            "config": {
                "model_sample_rate": 16000,
                "model_ft_frame_rate": 250,
                "n_seconds": 4,
            }
        },
        "state_dict": {},
    }


def test_from_checkpoint_forwards_cuda_index_to_model_and_loads_on_cpu(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.ckpt"
    models = []
    load_kwargs = {}

    def build_model(config):
        model = ModelDouble(config)
        models.append(model)
        return model

    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(api, "GuitarControlModel", build_model)
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: load_kwargs.update(kwargs) or _checkpoint_payload())
    monkeypatch.setattr(api.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api.torch.cuda, "device_count", lambda: 2)

    api.GuitarSynthesizer.from_checkpoint(device="cuda:1")
    assert models[0].device == torch.device("cuda:1")
    assert load_kwargs == {"map_location": "cpu"}


def test_session_load_and_cache_info_share_resolver_with_opposite_download_policy(monkeypatch, tmp_path):
    calls = []
    expected = tmp_path / "unified.ckpt"
    models = []

    def fake_resolve(path, cache_dir, *, allow_download=True):
        calls.append((path, cache_dir, allow_download))
        return expected

    monkeypatch.setattr(api, "resolve_checkpoint", fake_resolve)
    monkeypatch.setattr(api, "GuitarControlModel", lambda config: models.append(ModelDouble(config)) or models[-1])
    monkeypatch.setattr(api.torch, "load", lambda *args, **kwargs: _checkpoint_payload())
    session = GuitarSynthSession("weights.ckpt", cache_dir="cache")
    session.load()
    assert session.cache_info() == {"checkpoint": expected, "cached": True, "error": None}
    assert calls == [
        ("weights.ckpt", "cache", True),
        ("weights.ckpt", "cache", False),
    ]


def test_resolver_local_only_flag_is_forwarded(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        checkpoints,
        "hf_hub_download",
        lambda **kwargs: seen.update(kwargs) or str(tmp_path / "unified.ckpt"),
    )
    checkpoints.resolve_checkpoint(cache_dir=str(tmp_path), allow_download=False)
    assert seen["local_files_only"] is True
