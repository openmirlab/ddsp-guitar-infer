# ddsp-guitar-infer -- CLAUDE.md

## Scope

ddsp-guitar-infer is an inference-only PyTorch reimplementation of a
DDSP-based (Differentiable Digital Signal Processing) string-wise guitar
synthesizer: it renders a MIDI guitar performance to audio, one guitar
string (voice) at a time, using a control RNN + additive harmonic synthesis
+ filtered noise synthesis + reverb. See README.md for the public API and
CLI.

## Weights provenance (load-bearing -- read before touching checkpoints.py or LICENSE)

The pretrained checkpoint is downloaded from **`erl-j/ddsp-guitar-unified`**
on Hugging Face Hub (`unified.ckpt`, ~360 MB) -- a **personal, third-party
account**, not an openmirlab-controlled repo. As of this writing the HF
repo's own license field reports **"unknown"** and its README/model card is
empty: there is no explicit license grant for the weights themselves beyond
public availability on the Hub.

This package's own source code is Apache-2.0 (see LICENSE) and is an
original reimplementation, not copied from any DDSP reference
implementation -- the README's Acknowledgments section credits the
DDSP research (Engel et al., 2020) for the architectural approach only.
**The code license does not extend to the weights.** Do not assume the
checkpoint is safe to redistribute, fine-tune commercially, or bundle
without checking with the upstream author first.

Follow-up before wider distribution of the weights: either (a) get
explicit licensing terms from `erl-j`, or (b) mirror the checkpoint under
an openmirlab-controlled HF repo with an explicit license attached, and
point `DEFAULT_REPO` in `utils/checkpoints.py` at that mirror.

## Determinism (load-bearing -- read before writing golden-output tests)

Synthesis is **not deterministic by default**. `FilteredNoiseSynth.forward`
(`ddsp_guitar_utils/synth.py`) draws fresh noise from `torch.rand` on every
call, and `Reverb`/`MultiChannelReverb` seed their impulse-response
parameter from `torch.rand` at construction (overwritten by the loaded
checkpoint's `state_dict`, so this second source doesn't matter once
weights are loaded). Two renders of the same MIDI file will differ unless
the caller calls `torch.manual_seed(seed)` immediately before each
`render_midi(...)` call. `tests/test_e2e_regression.py` verifies this
property directly (same-seed match, different-seed mismatch) rather than
assuming it.

## Module layout

- `api.py` -- `GuitarSynthesizer`: checkpoint loading, device/dtype setup,
  windowed inference with cross-fade blending, chunked audio rendering.
- `cli.py` -- argparse CLI wrapping `api.load_synth` / `render_midi`.
- `model.py` -- `GuitarControlModel`: the control network (string/feature
  embeddings -> `SARNNBlock` stack -> classification+regression heads) plus
  `.render()`, which hands parameters to the DDSP synth.
- `utils/checkpoints.py` -- `resolve_checkpoint`: local path ->
  `DDSP_GUITAR_WEIGHTS` env var -> Hugging Face download, in that order.
  The only place network access happens.
- `utils/midi.py` -- MIDI -> per-string pitch/velocity frame grids, note
  extension/cropping, and the linear cross-fade blend used to stitch
  overlapping inference windows back together.
- `utils/preprocessing.py` -- raw MIDI/hex features -> normalized model
  input tensors.
- `ddsp_guitar_utils/synthesis_model.py` -- `DDSPModel`: harmonic + noise
  synthesis + reverb, folding the string dimension into batch around the
  underlying synth ops. Also carries unused decoder variants kept for
  reference (`DDSPDecoderAdapter`, `MixFcDecoder`) -- not on the active
  `GuitarControlModel` path.
- `ddsp_guitar_utils/synth.py` -- the actual DSP: additive harmonic
  synthesis, filtered noise synthesis, reverb (see Determinism above).
- `ddsp_guitar_utils/dsp.py` -- unit conversions, fold/unfold, `Quantizer`.
- `ddsp_guitar_utils/control_blocks.py` -- `SARNNBlock` (self-attention +
  bidirectional RNN), the repeated block in the control network.
- `ddsp_guitar_utils/nn.py` + `ddsp_guitar_utils/glotnet_wavenet/` --
  WaveNet-style conv stack used only by the unused `MixFcDecoder` path.

## Testing

- `tests/test_cli.py`, `tests/test_midi_utils.py` -- fast unit tests, no
  checkpoint needed, run in every CI job.
- `tests/test_e2e_regression.py` -- full pipeline (real checkpoint, real
  synthesis). Skips via `skipif` unless `DDSP_GUITAR_RUN_E2E=1` is set
  (CI leaves it unset, so these report skipped, not failed -- see
  `.github/workflows/test.yml`). Set `DDSP_GUITAR_WEIGHTS=/path/to/unified.ckpt`
  alongside it to avoid a live Hugging Face download.

## File-top header convention

Every load-bearing module starts with a header of this shape (as the
module docstring):

```python
"""<filename> -- <one-line role> (optionally marked as load-bearing).

2-3 sentences: what this file does, the key pattern, why this shape.

Reads: <key dependencies>
"""
```

Skip thin files (`__init__.py` re-export barrels, tiny primitives). Header
updates ride the same commit as changes to a file's role or main imports --
a stale header is a lie the next reader (human or agent) inherits.
