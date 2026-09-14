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
2. Regression: the render for a fixed seed + fixed MIDI is compared
   against a stored golden float array within a documented tolerance —
   see GOLDEN_TORCH_VERSION / CURRENT-torch measurement note below for
   why this is a tolerance, not an exact-digest match (org constitution
   article 2: "Bit-exact fixtures record their environment and guard on
   it" — float digests are only valid on the recording torch build).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
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

# The golden fixture was captured on torch 2.9.1 / CPU / erl-j/ddsp-guitar-unified
# unified.ckpt (sha256-pinned in config/checkpoints.toml). Re-measured
# 2026-09-14 against torch 2.13.0+cu130 (this repo's current floor and
# phonon's pinned torch) using the identical checkpoint bytes and seed:
# max abs diff = 6.045447662472725e-06 (float32; signal peak ~0.0786, so
# ~7.7e-5 relative), mean abs diff = 4.98e-08, out of 206399 samples
# 31570 were bit-identical and the rest differed only at this tiny scale
# (no structural divergence -- e.g. no sign flips, no order-of-magnitude
# jumps). This matches the org's documented pattern of small
# compounding-float-op drift across torch/BLAS builds (same class as
# madmom-infer's ULP-margin convention). GOLDEN_ATOL below is set to
# ~4.1x the measured max abs diff (2.5e-05), the same "~4x observed"
# margin convention other packages in this org use.
GOLDEN_TORCH_VERSION = "2.9.1+cpu"
GOLDEN_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "render_seed1234_torch291.npy"
GOLDEN_ATOL = 2.5e-05


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
    """Render must be close to the torch-2.9.1-recorded golden array.

    Not an exact-digest match: torch/BLAS build differences produce tiny
    (~1e-6 absolute) floating-point drift even with an identical checkpoint,
    seed, and code path -- reproduced and measured directly (see the module
    docstring / GOLDEN_ATOL comment above) rather than assumed. A failure
    here beyond GOLDEN_ATOL means either the golden needs re-recording after
    an intentional synthesis change, or a real regression was introduced.
    """
    torch.manual_seed(SEED_A)
    audio = synth.render_midi(midi_path)

    golden = np.load(GOLDEN_FIXTURE_PATH)
    actual = audio.numpy()
    assert actual.shape == golden.shape
    max_abs_diff = np.abs(actual.astype(np.float64) - golden.astype(np.float64)).max()
    assert max_abs_diff <= GOLDEN_ATOL, (
        f"render diverged from the torch {GOLDEN_TORCH_VERSION} golden fixture by "
        f"{max_abs_diff:.3e} (tolerance {GOLDEN_ATOL:.3e}, current torch "
        f"{torch.__version__}) -- see this test's docstring."
    )
