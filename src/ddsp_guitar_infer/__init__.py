"""Public API for ddsp-guitar-infer."""

from importlib import metadata

from .api import GuitarSynthesizer, GuitarSynthSession, load_synth

try:
    __version__ = metadata.version("ddsp-guitar-infer")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/unbuilt checkout
    __version__ = "0.0.0"

__all__ = ["GuitarSynthesizer", "GuitarSynthSession", "load_synth", "__version__"]
