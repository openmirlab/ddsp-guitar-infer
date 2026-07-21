# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `GuitarSynthSession`: an additive lifecycle API around `load_synth`, with
  idempotent loading, explicit release/terminal close, context-manager use,
  ready-only MIDI rendering, and download-free checkpoint cache inspection.
- `.github/workflows/test.yml`: CI matrix extended from `["3.10", "3.12"]` to
  `["3.10", "3.11", "3.12", "3.13"]` -- all four verified green locally before
  landing (`requires-python = ">=3.10"` already made no over-claim, so this
  closes an under-tested gap rather than raising the floor).
- `.github/workflows/test.yml`: new `build-smoke-test` job (org constitution
  art.7) -- `uv build` produces a wheel *from the sdist* (not the working
  tree), installs it into a clean venv, and imports `GuitarSynthesizer` /
  `load_synth` plus a CLI `--help` smoke test. Catches the empty-wheel class
  of packaging bug that `hatch build` alone can miss.

### Changed
- `pyproject.toml` dependency floors bumped after per-version local
  verification (Python 3.10/3.11/3.12/3.13, `pytest -q`: 3 passed / 3 skipped
  on every version):
  - `numpy`: `>=1.24,<3.0` -> `>=2.2.0` (no ceiling). NumPy 3.0 has not been
    released (latest is 2.5.x) and the old `<3.0` ceiling carried no written
    justification or tracking issue, so per art.3 ("floors, not ceilings") it
    is removed rather than kept as unexamined caution. The floor is bumped to
    `2.2.0`, not the newest `2.5.x` line, because `numpy>=2.3.0` requires
    Python >= 3.11 and this repo's CI matrix keeps Python 3.10 -- `2.2.x` is
    the newest line that still supports 3.10. (`uv`'s resolver picks `2.2.6`
    on 3.10 and `2.3.5`+ on 3.11+ from this same unbounded floor.)
  - `torch`: `>=2.4.0` -> `>=2.13.0`; `torchaudio`: `>=2.4.0` -> `>=2.11.0`.
    Both verified to support Python 3.10-3.13.
  - `huggingface_hub`: `>=0.24.0` -> `>=1.0.0`. huggingface_hub 1.0 (released
    2025-10-27) removes several long-deprecated parameters, including
    `hf_hub_download(local_dir_use_symlinks=...)`, which this package passed
    explicitly -- see the `checkpoints.py` fix below. `local_dir_use_symlinks`
    has been a documented "deprecated arg" (no functional effect once
    `local_dir` is set) since at least 0.24, so dropping it is a pure
    behavior-preserving cleanup on every version from 0.24 through 1.23, not
    a compromise made to reach 1.x.
- `src/ddsp_guitar_infer/utils/checkpoints.py`: dropped the
  `local_dir_use_symlinks=False` argument from the `hf_hub_download` call --
  required for `huggingface_hub>=1.0` (the parameter no longer exists there)
  and a no-op on the older versions this package still supports.
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
