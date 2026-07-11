import pretty_midi
import torch

from ddsp_guitar_infer.utils import midi as midi_utils


def test_pretty_midi_to_pitch_vel_produces_string_map():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=24)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)

    pitch, velocity = midi_utils.pretty_midi_to_pitch_vel(pm, frame_rate=50, legato=False)

    assert pitch.shape[0] == 6
    assert (pitch > 0).any()
    string_index = midi_utils.PITCH_TO_STRING[60]
    assert torch.all(pitch[string_index, :25] == 60)
    assert torch.all(velocity[string_index, :25] > 1.0)


def test_linear_interpolate_list_blends_chunks():
    a = torch.zeros(1, 6, 10, 1)
    b = torch.ones(1, 6, 10, 1)
    merged = midi_utils.linear_interpolate_list([a, b], skip_frames=4)

    assert merged.shape[2] == 16
    # Front region matches first chunk, tail matches second
    assert torch.allclose(merged[:, :, 0:6], torch.zeros_like(merged[:, :, 0:6]))
    assert torch.allclose(merged[:, :, -6:], torch.ones_like(merged[:, :, -6:]))
