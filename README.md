# ddsp-guitar-infer

**Inference-only PyTorch toolkit for DDSP-based string-wise guitar synthesis**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.13+](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)

## Why this exists

[DDSP (Differentiable Digital Signal Processing)](https://github.com/magenta/ddsp)
(Engel et al., ICLR 2020, Google Magenta) established the architectural
approach behind this package — combining classic DSP building blocks
(additive harmonic synthesis, filtered-noise synthesis) with neural control,
so a network only has to predict *synthesis parameters* rather than raw
audio. The original `magenta/ddsp` library is TensorFlow-based and is a
general-purpose research toolkit, not a guitar-specific, install-and-run
inference package.

ddsp-guitar-infer is a from-scratch PyTorch reimplementation targeted
specifically at MIDI-to-audio string-wise guitar synthesis: it renders a
MIDI guitar performance to audio, one string (voice) at a time, using a
control RNN plus additive harmonic + filtered-noise DDSP synthesis, with a
self-contained inference pipeline (no TensorFlow dependency) and an
auto-downloaded pretrained checkpoint.

## Acknowledgments

This package builds upon the DDSP framework by Google Magenta:

- **Upstream org**: [Google Magenta](https://github.com/magenta)
- **Authors**: Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, and Adam Roberts
- **Source project**: [magenta/ddsp](https://github.com/magenta/ddsp) — the foundational DDSP architecture and synthesis approach
- **Weights host**: pretrained checkpoint (`unified.ckpt`) hosted on Hugging Face Hub at [erl-j/ddsp-guitar-unified](https://huggingface.co/erl-j/ddsp-guitar-unified), a third-party personal account unaffiliated with Google Magenta or openmirlab (see [What this project will NEVER bundle](#what-this-project-will-never-bundle))

This package provides an inference-only PyTorch reimplementation for guitar synthesis. All credit for the core DDSP research belongs to the original authors.

## Citation

```bibtex
@inproceedings{engel2020ddsp,
  title={DDSP: Differentiable Digital Signal Processing},
  author={Engel, Jesse and Hantrakul, Lamtharn and Gu, Chenjie and Roberts, Adam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Features

- MIDI-to-audio guitar synthesis with per-string modeling
- Auto-download of pretrained checkpoint from Hugging Face (~unified.ckpt)
- Pitch correction: swap predicted f0 with MIDI pitch on active notes
- GPU acceleration with CPU fallback
- Self-contained DSP utilities (no TensorFlow dependency)

## Scope

**In scope:**
- MIDI-to-audio guitar synthesis via the pretrained `unified.ckpt` model, per-string additive + filtered-noise DDSP synthesis, pitch correction, chunked/windowed rendering

**Out of scope, forever:**
- Training or fine-tuning the model — this is an inference-only wrapper
- Instruments other than guitar (the model and checkpoint are guitar-specific)
- A TensorFlow code path — this package is a from-scratch PyTorch reimplementation and will not grow a TensorFlow dependency

## Install

```bash
uv sync
# or
pip install .
```

### Requirements

- **Python**: 3.10+
- **Core**: torch, torchaudio, numpy, scipy, soundfile, pretty_midi
- **Checkpoint**: Auto-downloaded from Hugging Face on first run (~360 MB,
  cached by `huggingface_hub`). To skip the download (e.g. in CI or
  offline), set `DDSP_GUITAR_WEIGHTS=/path/to/unified.ckpt` or pass
  `--checkpoint`/`checkpoint=` explicitly.

### Determinism

Synthesis uses a filtered-noise DDSP component seeded from PyTorch's
global RNG on every render, so **two renders of the same MIDI are not
identical by default**. For reproducible output, call
`torch.manual_seed(seed)` immediately before `render_midi(...)`.

## Quick Start

```python
from ddsp_guitar_infer import load_synth

synth = load_synth(device="cuda")
audio = synth.render_midi("song.mid")
synth.save_wav(audio, "song.wav")
```

```bash
# CLI
ddsp-guitar-infer song.mid --output song.wav --device cuda:0
```

## Usage

### CLI

```bash
# Basic synthesis
ddsp-guitar-infer song.mid --output song.wav

# GPU
ddsp-guitar-infer song.mid --output song.wav --device cuda:0

# Options
ddsp-guitar-infer song.mid \
    --output song.wav \
    --crop-seconds 30 \
    --no-pitch-correction \
    --legato \
    --render-chunk-seconds 8
```

### Python API

```python
from ddsp_guitar_infer import load_synth

synth = load_synth(device="cuda")
audio = synth.render_midi("song.mid")
synth.save_wav(audio, "song.wav")
```

For an explicit model lifecycle, use `GuitarSynthSession`.  `load()` is
idempotent; `release()` frees the resident model but retains downloaded
weights; and `close()` is terminal. `infer()` and `render_midi()` require a
loaded session.

```python
from ddsp_guitar_infer import GuitarSynthSession

with GuitarSynthSession(device="cuda:1") as session:
    audio = session.render_midi("song.mid")

# This only inspects the same configured checkpoint location; it never downloads.
cache = GuitarSynthSession(cache_dir="weights").cache_info()
```

## What this project will NEVER bundle

The pretrained checkpoint (~360 MB) is **never bundled in the package and
never git-tracked** — it is downloaded from Hugging Face Hub on first use
and cached locally by `huggingface_hub` (see [Requirements](#requirements)
above for how to point at a local file instead and skip the download
entirely, e.g. in CI or offline).

The checkpoint (`erl-j/ddsp-guitar-unified` on Hugging Face) is hosted under
a **third-party personal account**, not an openmirlab repo. Its Hugging
Face license field is currently **unspecified ("unknown")** and it has no
model card. This package's own code is Apache-2.0 (see [LICENSE](LICENSE))
and is an original reimplementation, not copied from any DDSP reference
implementation — the [Acknowledgments](#acknowledgments) section above
credits the DDSP research (Engel et al., 2020) for the architectural
approach only. **The code license does not extend to the weights.** Treat
redistribution or commercial fine-tuning of the checkpoint as unresolved
until upstream licensing is clarified or the checkpoint is mirrored under
an openmirlab-controlled repo with explicit terms — see
[CLAUDE.md](CLAUDE.md) for the concrete follow-up options.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
```

Fast unit tests (`tests/test_cli.py`, `tests/test_midi_utils.py`) need no
checkpoint and run in every CI job. The full pipeline regression
(`tests/test_e2e_regression.py`) needs the real ~360 MB checkpoint and
self-skips unless `DDSP_GUITAR_RUN_E2E=1` is set — CI leaves it unset, so
CI never downloads weights or hits the network. See
[CLAUDE.md](CLAUDE.md) for the full testing philosophy, module layout, and
file-header convention used across this codebase.

## License

Apache-2.0

## Support

For bugs, feature requests, or questions, please open an issue on [GitHub](https://github.com/openmirlab/ddsp-guitar-infer/issues).
