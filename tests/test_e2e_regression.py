"""test_e2e_regression.py — full MIDI-to-audio synthesis regression (opt-in).

Exercises the real pipeline end to end: build a short MIDI performance,
load the actual checkpoint (Hugging Face download or DDSP_GUITAR_WEIGHTS
override), and render audio. Skipped by default (no network access /
~360MB checkpoint download in ordinary CI) — opt in locally or in a
weights-provisioned job with DDSP_GUITAR_RUN_E2E=1.

Two properties are checked:
1. Determinism: FilteredNoiseSynth draws from torch.rand every call, so
   the model is only reproducible across runs if torch's global RNG is
   seeded beforehand (see ddsp_guitar_utils/synth.py header). Same seed
   must give a bit-identical digest; a different seed must not.
2. Regression: the digest for a fixed seed + fixed MIDI is pinned below.
   If it changes, either the synthesis code changed on purpose (update
   EXPECTED_DIGEST_SEED_1234) or a regression was introduced.
"""

from __future__ import annotations

import hashlib
import os

import pretty_midi
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("DDSP_GUITAR_RUN_E2E") != "1",
    reason=(
        "End-to-end synthesis needs the real checkpoint (Hugging Face download "
        "or DDSP_GUITAR_WEIGHTS) and is slow; opt in with DDSP_GUITAR_RUN_E2E=1"
    ),
)

SEED_A = 1234
SEED_B = 9999

# Captured on: torch 2.9.1 / CPU / erl-j/ddsp-guitar-unified unified.ckpt.
# A change here after an intentional model/DSP change is expected; an
# unexplained change signals a regression.
EXPECTED_DIGEST_SEED_1234 = (
    "87a9c0f20dde2c9be3a5129201e8f0c0d0551b763d703399b55654a92841a3d7"
)


def _build_test_midi() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25)  # steel-string acoustic guitar
    notes = [
        (60, 0.0, 0.5, 90),
        (64, 0.5, 1.0, 85),
        (67, 1.0, 1.5, 95),
        (72, 1.5, 2.2, 100),
    ]
    for pitch, start, end, velocity in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    return pm


def _digest(audio: torch.Tensor) -> str:
    return hashlib.sha256(audio.numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def synth():
    from ddsp_guitar_infer import load_synth

    return load_synth(device="cpu")


@pytest.fixture(scope="module")
def midi_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("e2e") / "test_notes.mid"
    _build_test_midi().write(str(path))
    return path


def test_render_is_deterministic_given_a_seed(synth, midi_path):
    torch.manual_seed(SEED_A)
    audio_a = synth.render_midi(midi_path)
    torch.manual_seed(SEED_A)
    audio_b = synth.render_midi(midi_path)

    assert _digest(audio_a) == _digest(audio_b)


def test_render_differs_across_seeds(synth, midi_path):
    torch.manual_seed(SEED_A)
    audio_a = synth.render_midi(midi_path)
    torch.manual_seed(SEED_B)
    audio_b = synth.render_midi(midi_path)

    assert _digest(audio_a) != _digest(audio_b)


def test_render_regression_fixture(synth, midi_path):
    torch.manual_seed(SEED_A)
    audio = synth.render_midi(midi_path)

    assert _digest(audio) == EXPECTED_DIGEST_SEED_1234
