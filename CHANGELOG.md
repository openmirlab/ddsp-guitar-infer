# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Standalone git identity: the project previously sat untracked inside an
  unrelated monorepo. First commit captures the clean source tree only
  (`.venv/`, `.pytest_cache/`, `__pycache__/`, build artifacts excluded).
- `DDSP_GUITAR_WEIGHTS` environment variable
  (`src/ddsp_guitar_infer/utils/checkpoints.py`): points `resolve_checkpoint`
  at a local `.ckpt` file, bypassing the Hugging Face download. Lets CI and
  offline use skip the ~360 MB `erl-j/ddsp-guitar-unified` download entirely.
- `tests/test_e2e_regression.py`: an opt-in (`DDSP_GUITAR_RUN_E2E=1`)
  end-to-end regression fixture. Renders a short built-in MIDI performance
  with a fixed `torch.manual_seed`, and checks (a) same-seed renders are
  bit-identical, (b) different seeds differ, (c) the digest for the pinned
  seed matches a recorded fixture value. Confirms synthesis is
  **not** deterministic by default -- `FilteredNoiseSynth` and the reverb
  impulse-response init draw from `torch.rand`, so reproducibility requires
  the caller to seed torch's global RNG before rendering.
- `.github/workflows/test.yml` + `.github/workflows/publish.yml`: CI now
  runs the unit test suite on every push/PR (previously no CI existed), and
  PyPI publishing is gated on the test suite passing first.
- File-top navigability headers on all load-bearing modules under
  `src/ddsp_guitar_infer/` (api, cli, model, utils/*, ddsp_guitar_utils/*),
  including a note on `synth.py` about the RNG/determinism caveat above.
- `CLAUDE.md` documenting module layout, weights provenance, and the
  determinism caveat for future contributors/agents.
- `__version__` on the package (`ddsp_guitar_infer.__version__`), sourced
  from installed package metadata so `pyproject.toml`'s `version` stays the
  single source of truth.
- `[project.urls]` (Homepage/Repository/Changelog) in `pyproject.toml`.

### Changed
- `pyproject.toml` author changed from the generic "worzpro Development
  Team" to a real contact (`Paul <bernie40916@gmail.com>`); added
  `license-files = ["LICENSE"]`.

### Documentation
- README: recorded that the pretrained checkpoint
  (`erl-j/ddsp-guitar-unified` on Hugging Face) is hosted under a
  **third-party personal account with no stated license** ("unknown" per
  the repo's own license field, no model card). This package's own code
  remains Apache-2.0; the weights carry no license grant from their author
  beyond public availability on the Hub. Follow-up: either obtain explicit
  licensing terms from the upstream author or mirror the checkpoint under
  an openmirlab-controlled Hugging Face repo with an explicit license
  before treating redistribution of the weights themselves as clear.

## [0.1.0] - 2024

### Added
- Initial inference-only PyTorch reimplementation of the DDSP string-wise
  guitar control network and synthesizer.
- `GuitarSynthesizer` / `load_synth` high-level API and `ddsp-guitar-infer`
  CLI for MIDI-to-audio rendering.
- Auto-download of the pretrained checkpoint from Hugging Face Hub.
- Per-string harmonic + filtered-noise DDSP synthesis with per-voice
  reverb, windowed/chunked rendering for long MIDI files.
