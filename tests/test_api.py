import torch

from ddsp_guitar_infer.api import _resolve_device


def test_resolve_device_auto_matches_none():
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert _resolve_device("auto") == expected
    assert _resolve_device(None) == expected


def test_resolve_device_passes_through_explicit_strings():
    assert _resolve_device("cpu") == torch.device("cpu")
