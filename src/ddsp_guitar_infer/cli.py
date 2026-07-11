"""cli.py — command-line entry point (★ load-bearing).

argparse wrapper around api.load_synth/render_midi; exposes device,
checkpoint override, crop/legato/pitch-correction and chunked-rendering
flags. Wired as the `ddsp-guitar-infer` console script in pyproject.toml.

Reads: api.load_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ddsp_guitar_infer.api import load_synth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DDSP Guitar - render MIDI to neural audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("midi", type=str, help="Input MIDI file")
    parser.add_argument("--output", type=str, help="Output WAV path (defaults to <midi>.wav)")
    parser.add_argument("--checkpoint", type=str, help="Path to a .ckpt file (defaults to HF download)")
    parser.add_argument("--cache-dir", type=str, help="Optional cache directory for downloads")
    parser.add_argument("--device", type=str, help="Device string such as cuda:0 or cpu")
    parser.add_argument("--crop-seconds", type=int, help="Trim MIDI to the first N seconds before rendering")
    parser.add_argument("--skip-ratio", type=float, default=0.5, help="Window overlap ratio (0-1)")
    parser.add_argument("--legato", action="store_true", help="Keep overlapping notes on each string")
    parser.add_argument(
        "--pitch-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace predicted pitch with MIDI pitch on active notes",
    )
    parser.add_argument(
        "--render-chunk-seconds",
        type=float,
        default=8.0,
        help="Render audio in chunks of this many seconds to reduce memory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    midi_path = Path(args.midi)
    if not midi_path.exists():
        parser.error(f"Input MIDI not found: {midi_path}")

    output = Path(args.output) if args.output else midi_path.with_suffix(".wav")

    synth = load_synth(checkpoint=args.checkpoint, device=args.device, cache_dir=args.cache_dir)
    synth.render_midi(
        midi_path=midi_path,
        output_path=output,
        crop_seconds=args.crop_seconds,
        legato=args.legato,
        pitch_correction=args.pitch_correction,
        skip_ratio=args.skip_ratio,
        render_chunk_seconds=args.render_chunk_seconds,
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
