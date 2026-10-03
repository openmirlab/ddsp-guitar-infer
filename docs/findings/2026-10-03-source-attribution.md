# DDSP-Guitar source attribution preflight — 2026-10-03

Before changing package metadata, compare this package with the official
[`erl-j/ddsp-guitar`](https://github.com/erl-j/ddsp-guitar) source at commit
`0a568e8d8bdf10b1b33a50f5cf8826365f76df45`. The upstream GitHub API
reports Apache-2.0 and that commit contains `LICENSE`. Its README identifies
the code as the demo for Jonason et al.'s guitar synthesis paper; Magenta DDSP
is an acknowledged architectural predecessor, not the guitar checkpoint's
author. The paper lists Nicolas Jonason, Xin Wang, Erica Cooper, Lauri Juvela,
Bob L. T. Sturm, and Junichi Yamagishi.

All six files in `src/ddsp_guitar_infer/ddsp_guitar_utils/glotnet_wavenet/`
(`__init__.py`, `activations.py`, `convolution.py`,
`convolution_layer.py`, `convolution_stack.py`, `wavenet.py`) are byte-for-byte
identical to the same-named upstream files at that commit. This was checked
with `Path.read_bytes()` equality; `activations.py` itself carries
`Copyright 2022 Lauri Juvela` and the Apache-2.0 notice. Several other
files retain substantial upstream code: a raw `difflib.SequenceMatcher`
comparison returned approximately 0.95 for `nn.py`, 0.98 for `synth.py`,
0.97 for `synthesis_model.py`, and 0.82 for `preprocessing.py`. Those ratios
are navigation evidence, not a legal similarity metric; inspect the actual
diff before asserting an exact relationship for each adapted file.

The current README and CLAUDE.md say the package is an original
reimplementation with no code copied from a reference implementation.
That blanket claim is false for the six exact files. Correct it, credit the
guitar source and the separate Magenta foundation, retain the upstream
copyright notice, and add a NOTICE for the shipped code. This finding concerns
the **code**. The hosted `unified.ckpt` remains license `NOASSERTION`; the
upstream code's Apache-2.0 license does not grant terms for that separate
weights artifact.
