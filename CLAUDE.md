# ddsp-guitar-infer -- CLAUDE.md

## Scope

ddsp-guitar-infer is an inference-only PyTorch reimplementation of a
DDSP-based (Differentiable Digital Signal Processing) string-wise guitar
synthesizer: it renders a MIDI guitar performance to audio, one guitar
string (voice) at a time, using a control RNN + additive harmonic synthesis
+ filtered noise synthesis + reverb. See README.md for the public API and
CLI.

**Status**: Apache-2.0, PyPI-publishable code (see `.github/workflows/publish.yml`)
— but the pretrained checkpoint it downloads carries **`NOASSERTION`**
(verified, not merely "unknown" by omission -- see Weights license below),
so redistribution/commercial use of the *weights* specifically is not
cleared. Weights are externalized to a Hugging-Face-backed
download-on-demand cache, never bundled or git-tracked (org constitution
article 4).

## Weights provenance (load-bearing -- read before touching checkpoints.py or LICENSE)

The pretrained checkpoint is downloaded from **`erl-j/ddsp-guitar-unified`**
on Hugging Face Hub (`unified.ckpt`, ~360 MB) -- a **personal, third-party
account**, not an openmirlab-controlled repo.

This package's own source code is Apache-2.0 (see LICENSE) and is an
original reimplementation, not copied from any DDSP reference
implementation -- the README's Acknowledgments section credits the
DDSP research (Engel et al., 2020) for the architectural approach only.
**The code license does not extend to the weights.** Do not assume the
checkpoint is safe to redistribute, fine-tune commercially, or bundle
without checking with the upstream author first.

Follow-up before wider distribution of the weights: either (a) get
explicit licensing terms from `erl-j` (Nicolas Jonason, per the arXiv
listing below), or (b) mirror the checkpoint under an
openmirlab-controlled HF repo with an explicit license attached, and
point `DEFAULT_REPO` in `utils/checkpoints.py` at that mirror. This is
still open (tracked org-wide in `openmirlab-dev/weights.md`'s "queued"
row for this package) -- neither the sha256 pin nor the license
verification below resolves the personal-account hosting risk.

### Weights license: NOASSERTION (verified 2026-09-14, primary sources)

No party in the chain grants an explicit license for the checkpoint
itself. Evidence, checked in this order:

1. **The HF model repo's own metadata** (the authoritative record for
   *this specific artifact*):
   `curl https://huggingface.co/api/models/erl-j/ddsp-guitar-unified` ->
   `"cardData": {"license": "unknown"}`, `"tags": ["license:unknown", ...]`.
   The repo's `README.md` is a bare 3-line YAML front-matter stub (`license:
   unknown`) with no prose, no model card.
2. **The author's companion code repo**,
   [github.com/erl-j/ddsp-guitar](https://github.com/erl-j/ddsp-guitar):
   `gh api repos/erl-j/ddsp-guitar --jq .license` -> `apache-2.0`
   (confirmed `LICENSE` file present, standard Apache License 2.0 text).
   This repo is the **official demo code for the paper this checkpoint
   comes from** -- its own README states it is demo code for
   [*DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance
   from String-wise MIDI Input*](https://arxiv.org/abs/2309.07658)
   (Jonason, Wang, Cooper, Juvela, Sturm, Yamagishi; arXiv:2309.07658,
   2023-09-14; confirmed via arXiv's own citation metadata, real author
   list -- `erl-j` is Nicolas Jonason, per the GitHub account's linked
   name), and its `render_midi.py` downloads **this exact file** from
   **this exact URL** (`https://huggingface.co/erl-j/ddsp-guitar-unified/
   resolve/main/unified.ckpt`) as "the *unified* model" -- confirming this
   checkpoint is that paper's official released artifact, not a
   coincidentally-similarly-named file.
3. **The gap**: Apache-2.0 in (2) licenses that GitHub repo's *code*. It
   does not, on its own, constitute a license grant for a large binary
   artifact hosted in a *separate* HF repo whose own license field says
   "unknown" rather than inheriting or restating Apache-2.0. Nothing in
   either repo, the README, or the paper's abstract page states the
   trained weights are released under the same terms as the code. Per
   article 3 ("verify upstream licenses via API/file, never via claims")
   this package does not infer a grant that was not made explicit --
   despite the strong same-author, same-artifact circumstantial link,
   `license = "NOASSERTION"` is the honest, verifiable record.

**Consequence**: redistribution, re-hosting, and commercial use/fine-tuning
of `unified.ckpt` are all unresolved until `erl-j`/Nicolas Jonason confirms
terms explicitly. This package only downloads the checkpoint at runtime
into the end user's own cache (never bundles or redistributes it) and
verifies its integrity, which does not require a redistribution license.
See README's Scope and "What this project will NEVER bundle" sections for
the user-facing statement of the same finding.

### Integrity: pinned, not an exception (2026-09-14)

`config/checkpoints.toml` records a real `sha256`/`size_bytes`/`revision`
for `unified.ckpt`, not the article-4 "unavailable" marker. This was a
deliberate pin, not the default: `erl-j/ddsp-guitar-unified` turned out to
be a genuinely stable, single-commit HF repo (created and last modified
2023-12-23, one branch, one target commit). Two independent measurements
of the file agreed exactly -- the Hub API's own reported git-lfs object
hash (`.../api/models/erl-j/ddsp-guitar-unified/paths-info/main`) and an
independently downloaded copy hashed locally with `sha256sum` -- both
`f90db7735df5be6be389a626d568d2a9928759ebfa192a516ccefee928b4ded7`
(359,321,113 bytes). `revision` pins that exact 40-hex commit sha rather
than the mutable `main` branch ref, so the record keeps describing this
artifact even if the author pushes new commits later.

`utils/checkpoints.resolve_checkpoint` verifies this hash after every
hub-resolved download (`allow_download=True`, the path `load()`/
`GuitarSynthesizer.from_checkpoint` use) and raises `RuntimeError` naming
the expected/actual digests on mismatch. Two paths deliberately skip
verification, matching adtof-infer's precedent: an explicit `checkpoint=`
argument or `DDSP_GUITAR_WEIGHTS` override is assumed to be an
intentional, known-good file (a test fixture or a caller's own build) and
is never hashed; and `GuitarSynthSession.cache_info()`'s read-only
`allow_download=False` check must never hash a ~360 MB file just to
report cache status.

If a future revision bump ever needs re-pinning: re-run the same
cross-check (Hub API `paths-info` vs. a fresh independent download) before
trusting a new hash -- a single HF-hosted measurement alone is one source,
not two, since the API and the download both ultimately come from the
same host.

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
  `GuitarSynthSession` adds an explicit load/release/close lifecycle around
  the legacy `load_synth` API; its `cache_info()` uses local-only resolution
  and must never initiate a checkpoint download.
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

## Testing philosophy

- `tests/test_cli.py`, `tests/test_midi_utils.py` -- fast unit tests, no
  checkpoint needed, run in every CI job.
- `tests/test_e2e_regression.py` -- full pipeline (real checkpoint, real
  synthesis). Skips via `skipif` unless `DDSP_GUITAR_RUN_E2E=1` is set
  (CI leaves it unset, so these report skipped, not failed -- see
  `.github/workflows/test.yml`). Set `DDSP_GUITAR_WEIGHTS=/path/to/unified.ckpt`
  alongside it to avoid a live Hugging Face download. This is also the test
  that directly verifies the Determinism contract above (same-seed match,
  different-seed mismatch) rather than assuming it.

## Verification / test commands

```bash
uv sync --extra dev
uv run pytest -q                    # fast tests only (default CI job)
DDSP_GUITAR_RUN_E2E=1 uv run pytest -q   # + real-checkpoint e2e regression
uv run ruff check .
```

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
