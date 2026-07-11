"""MIDI conditioning helpers for ddsp-guitar inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import pretty_midi
import torch


STRING_TO_PITCH = {
    0: range(40, 45),
    1: range(45, 50),
    2: range(50, 55),
    3: range(55, 60),
    4: range(60, 65),
    5: range(65, 89),
}
PITCH_TO_STRING = {pitch: string for string, pitches in STRING_TO_PITCH.items() for pitch in pitches}


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    midi_path = Path(path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return pretty_midi.PrettyMIDI(str(midi_path))


def crop_pretty_midi(pm: pretty_midi.PrettyMIDI, seconds: float) -> pretty_midi.PrettyMIDI:
    for instrument in pm.instruments:
        for note in list(instrument.notes):
            if note.start > seconds:
                instrument.notes.remove(note)
            elif note.end > seconds:
                note.end = seconds
    return pm


def remove_out_of_range_notes(pm: pretty_midi.PrettyMIDI, min_pitch: int = 40, max_pitch: int = 88) -> pretty_midi.PrettyMIDI:
    pm.instruments = [inst for inst in pm.instruments if not inst.is_drum]
    for instrument in pm.instruments:
        for note in list(instrument.notes):
            if note.pitch < min_pitch or note.pitch > max_pitch:
                instrument.notes.remove(note)
    return pm


def extend_notes(pm: pretty_midi.PrettyMIDI, factor: float = 4.0) -> pretty_midi.PrettyMIDI:
    for instrument in pm.instruments:
        for note in instrument.notes:
            note.end = factor * (note.end - note.start) + note.start
    return pm


def midi_to_model_scale_velocity(midi_velocity: torch.Tensor, min_pseudo_vel=1.08, max_pseudo_vel=1.12) -> torch.Tensor:
    scaled = torch.where(
        midi_velocity == 0,
        torch.zeros_like(midi_velocity),
        min_pseudo_vel + (midi_velocity / 127.0) * (max_pseudo_vel - min_pseudo_vel),
    )
    scaled[midi_velocity == 0] = 0.0
    return scaled


def pretty_midi_to_pitch_vel(pm: pretty_midi.PrettyMIDI, frame_rate: int, legato: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    seconds = pm.get_end_time()
    frames = math.ceil(seconds * frame_rate)
    pitch = torch.zeros((6, frames))
    velocity = torch.zeros((6, frames))

    for instrument in pm.instruments:
        for note in instrument.notes:
            start = int(note.start * frame_rate)
            end = int(note.end * frame_rate)
            string = PITCH_TO_STRING.get(note.pitch)
            if string is None:
                continue
            if not legato and start > 1 and pitch[string, start - 1] != 0:
                pitch[string, start - 2:start] = 0
                velocity[string, start - 2:start] = 0
            pitch[string, start:end] = note.pitch
            velocity[string, start:end] = note.velocity

    velocity = midi_to_model_scale_velocity(velocity)
    return pitch, velocity


def linear_interpolate(A: torch.Tensor, B: torch.Tensor, skip_frames: int) -> torch.Tensor:
    batch, voice, time, features = A.shape
    weights = torch.linspace(0, 1, steps=skip_frames, device=A.device).view(1, 1, skip_frames, 1)
    A_overlap = A[:, :, -skip_frames:, :]
    B_overlap = B[:, :, :skip_frames, :]
    blended = A_overlap * (1 - weights) + B_overlap * weights
    A_head = A[:, :, :-skip_frames, :]
    B_tail = B[:, :, skip_frames:, :]
    return torch.cat([A_head, blended, B_tail], dim=2)


def linear_interpolate_list(chunks, skip_frames: int) -> torch.Tensor:
    output = chunks[0]
    for chunk in chunks[1:]:
        output = linear_interpolate(output, chunk, skip_frames)
    return output

__all__ = [
    "load_midi",
    "crop_pretty_midi",
    "remove_out_of_range_notes",
    "extend_notes",
    "pretty_midi_to_pitch_vel",
    "linear_interpolate_list",
]
