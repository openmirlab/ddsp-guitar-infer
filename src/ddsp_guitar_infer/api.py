"""api.py — high-level inference entry point (★ load-bearing).

Wraps GuitarControlModel with device/dtype setup, MIDI-to-audio rendering
in overlapping windows (skip_ratio) with cross-fade blending, and chunked
waveform rendering to bound memory. Resolves the checkpoint via
resolve_checkpoint (local path, DDSP_GUITAR_WEIGHTS env var, or a
Hugging Face download).

Reads: model.GuitarControlModel · utils.midi · utils.checkpoints.resolve_checkpoint ·
utils.preprocessing.preprocess_model_inputs · ddsp_guitar_utils.dsp.convert_dtype
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from tqdm import tqdm

from ddsp_guitar_infer.model import GuitarControlModel
from ddsp_guitar_infer.utils import midi as midi_utils
from ddsp_guitar_infer.utils.checkpoints import resolve_checkpoint
from ddsp_guitar_infer.utils.preprocessing import preprocess_model_inputs
from ddsp_guitar_infer.ddsp_guitar_utils.dsp import convert_dtype


def _resolve_device(device: Optional[str]) -> torch.device:
    """Resolve a device string, treating None and "auto" as auto-detect."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class GuitarSynthesizer:
    """Wrapper that loads the DDSP control model and renders MIDI to audio."""

    model: GuitarControlModel
    device: torch.device
    sample_rate: int
    frame_rate: int
    model_frames: int
    dtype: torch.dtype

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> "GuitarSynthesizer":
        ckpt_path = resolve_checkpoint(checkpoint, cache_dir)
        raw = torch.load(ckpt_path, map_location="cpu")
        config = raw["hyper_parameters"]["config"]
        model = GuitarControlModel(config)
        model.load_state_dict(raw["state_dict"], strict=True)

        dev = _resolve_device(device)
        model.to(dev)
        model.eval()

        return cls(
            model=model,
            device=dev,
            sample_rate=config["model_sample_rate"],
            frame_rate=config["model_ft_frame_rate"],
            model_frames=config["n_seconds"] * config["model_ft_frame_rate"],
            dtype=next(model.parameters()).dtype,
        )

    # ------------------------------------------------------------------
    def render_midi(
        self,
        midi_path: str | Path,
        output_path: Optional[str | Path] = None,
        crop_seconds: Optional[int] = None,
        legato: bool = False,
        pitch_correction: bool = True,
        skip_ratio: float = 0.5,
        render_chunk_seconds: Optional[float] = None,
    ) -> torch.Tensor:
        pm = midi_utils.load_midi(midi_path)
        if crop_seconds is not None:
            pm = midi_utils.crop_pretty_midi(pm, crop_seconds)
        pm = midi_utils.remove_out_of_range_notes(pm)
        pm = midi_utils.extend_notes(pm)

        pitch, velocity = midi_utils.pretty_midi_to_pitch_vel(pm, self.frame_rate, legato=legato)

        frames_before_padding = pitch.shape[-1]
        total_frames = ((frames_before_padding + self.model_frames - 1) // self.model_frames) * self.model_frames
        pad = total_frames - frames_before_padding
        if pad > 0:
            pitch = torch.nn.functional.pad(pitch, (0, pad))
            velocity = torch.nn.functional.pad(velocity, (0, pad))

        voice_index = torch.arange(6)[None, :, None].repeat(1, 1, total_frames)
        skip_frames = int(self.model_frames * skip_ratio)
        synth_chunks = []

        with torch.no_grad():
            for start in tqdm(range(0, total_frames, skip_frames), desc="generating synth params"):
                end = start + self.model_frames
                pitch_window = pitch[..., start:end]
                velocity_window = velocity[..., start:end]
                if pitch_window.shape[-1] < self.model_frames:
                    pad_width = self.model_frames - pitch_window.shape[-1]
                    pitch_window = torch.nn.functional.pad(pitch_window, (0, pad_width))
                    velocity_window = torch.nn.functional.pad(velocity_window, (0, pad_width))

                inputs = {
                    "midi_pitch": pitch_window.unsqueeze(0),
                    "midi_pseudo_velocity": velocity_window.unsqueeze(0),
                    "string_index": voice_index[:, :, start:end].unsqueeze(-1),
                }
                inputs = preprocess_model_inputs(inputs)
                inputs = convert_dtype(inputs, {torch.float32: self.dtype, torch.float64: self.dtype})
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                preds = self.model(inputs)
                if pitch_correction:
                    active = inputs["midi_pseudo_velocity"] != 1
                    preds["hex_f0_scaled_hat"][active] = inputs["midi_pitch_scaled"][active]
                chunk = {}
                for key, value in preds.items():
                    if isinstance(value, torch.Tensor):
                        chunk[key] = value.detach().cpu()
                    else:
                        chunk[key] = value
                synth_chunks.append(chunk)

        features = ["harmonic_partial_amp_output", "harmonic_global_amp_output", "noise_band_amp_output", "hex_f0_scaled_hat"]
        full_params = {}
        for key in features:
            tensors = [chunk[key] for chunk in synth_chunks]
            merged = midi_utils.linear_interpolate_list(tensors, skip_frames)
            full_params[key] = merged[:, :, :frames_before_padding, :]

        full_params["voice_index"] = voice_index[:, :, :frames_before_padding].unsqueeze(-1)
        chunk_frames = self._get_chunk_frames(render_chunk_seconds)
        target_samples = int(pm.get_end_time() * self.sample_rate)
        audio = self._render_in_chunks(full_params, frames_before_padding, chunk_frames, target_samples)

        if output_path is not None:
            self.save_wav(audio, output_path)

        return audio

    def _get_chunk_frames(self, render_chunk_seconds: Optional[float]) -> int:
        if render_chunk_seconds is None:
            return self.model_frames
        chunk_frames = max(int(render_chunk_seconds * self.frame_rate), self.frame_rate)
        return chunk_frames

    def _render_in_chunks(
        self,
        params: dict,
        total_frames: int,
        chunk_frames: int,
        target_samples: int,
    ) -> torch.Tensor:
        audio_chunks = []
        keys = [
            "harmonic_partial_amp_output",
            "harmonic_global_amp_output",
            "noise_band_amp_output",
            "hex_f0_scaled_hat",
            "voice_index",
        ]

        for start in tqdm(range(0, total_frames, chunk_frames), desc="rendering audio chunks"):
            end = min(start + chunk_frames, total_frames)
            chunk = {}
            for key in keys:
                value = params[key]
                chunk[key] = value[:, :, start:end, :]

            chunk_frames_count = end - start
            chunk_samples = int(round(chunk_frames_count * self.sample_rate / self.frame_rate))
            render_out = self.model.render(chunk, n_samples=chunk_samples)["output"]
            audio_chunks.append(render_out.detach().cpu().squeeze())

        audio = torch.cat(audio_chunks, dim=-1)
        return audio[:target_samples]

    def save_wav(self, audio: torch.Tensor, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio.numpy(), self.sample_rate)


def load_synth(
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> GuitarSynthesizer:
    return GuitarSynthesizer.from_checkpoint(checkpoint, device, cache_dir)


__all__ = ["GuitarSynthesizer", "load_synth"]
