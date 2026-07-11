# ddsp-guitar-infer

**Inference-only PyTorch toolkit for DDSP-based string-wise guitar synthesis**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

## Overview

ddsp-guitar-infer renders MIDI guitar performances to audio using a DDSP-based string-wise synthesizer. It models each guitar string independently with harmonic + noise components, producing realistic nylon/steel guitar tones from MIDI input.

Features:
- MIDI-to-audio guitar synthesis with per-string modeling
- Auto-download of pretrained checkpoint from Hugging Face (~unified.ckpt)
- Pitch correction: swap predicted f0 with MIDI pitch on active notes
- GPU acceleration with CPU fallback
- Self-contained DSP utilities (no TensorFlow dependency)

## Acknowledgments

This package builds upon the DDSP framework by Google Magenta:

- **[DDSP: Differentiable Digital Signal Processing](https://github.com/magenta/ddsp)** by Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, and Adam Roberts — The foundational DDSP architecture and synthesis approach.

This package provides an inference-only PyTorch reimplementation for guitar synthesis. All credit for the core DDSP research belongs to the original authors.

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

## Installation

```bash
uv sync
# or
pip install .
```

### Requirements

- **Python**: 3.10+
- **Core**: torch, torchaudio, numpy, scipy, soundfile, pretty_midi
- **Checkpoint**: Auto-downloaded from Hugging Face on first run

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

## Citation

```bibtex
@inproceedings{engel2020ddsp,
  title={DDSP: Differentiable Digital Signal Processing},
  author={Engel, Jesse and Hantrakul, Lamtharn and Gu, Chenjie and Roberts, Adam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## License

Apache-2.0

## Support

For bugs, feature requests, or questions, please open an issue on [GitHub](https://github.com/openmirlab/ddsp-guitar-infer/issues).
