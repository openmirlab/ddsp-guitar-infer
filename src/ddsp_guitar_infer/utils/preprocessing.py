import torch

from ddsp_guitar_infer.ddsp_guitar_utils.dsp import (
    scale_db,
    hz_to_unit,
    midi_to_unit,
    hz_to_midi,
)

GUITAR_F0_MIN_HZ = 35.0
GUITAR_F0_MAX_HZ = 1200.0
MIDI_MIN = float(hz_to_midi(GUITAR_F0_MIN_HZ))
MIDI_MAX = float(hz_to_midi(GUITAR_F0_MAX_HZ))

def preprocess_model_inputs(inputs):
    preprocessed_inputs = {}
    if "hex_loudness" in inputs:
        preprocessed_inputs["hex_loudness_scaled"] = scale_db(inputs["hex_loudness"])[...,None]
    if "hex_pitch" in inputs:
        preprocessed_inputs["hex_f0_scaled"] = hz_to_unit(inputs["hex_pitch"],GUITAR_F0_MIN_HZ,GUITAR_F0_MAX_HZ)[...,None]
    if "hex_periodicity" in inputs:
        preprocessed_inputs["hex_periodicity"] = inputs["hex_periodicity"][...,None]
    if "string_index" in inputs:
        preprocessed_inputs["voice_index"] = inputs["string_index"][...,None]
    # bug here??
    if "hex_centroid_scaled" in inputs:
        preprocessed_inputs["hex_centroid_scaled"] = inputs["hex_centroid_scaled"][...,None]
    if "hex_audio" in inputs:
        preprocessed_inputs["hex_audio"] = inputs["hex_audio"]

    # midi
    if "midi_pitch" in inputs:
        preprocessed_inputs["midi_pitch_scaled"] = midi_to_unit(
            inputs["midi_pitch"], MIDI_MIN, MIDI_MAX, clip=False
        )[...,None]
    if "midi_pseudo_velocity" in inputs:
        preprocessed_inputs["midi_pseudo_velocity"] = inputs["midi_pseudo_velocity"][...,None]
    if "midi_activity" in inputs:
        preprocessed_inputs["midi_activity"] = inputs["midi_activity"][...,None]
    if "midi_onsets" in inputs:
        preprocessed_inputs["midi_onsets"] = inputs["midi_onsets"][...,None]
    if "midi_offsets" in inputs:
        preprocessed_inputs["midi_offsets"] = inputs["midi_offsets"][...,None]
    if "midi_duration_since_previous_onset" in inputs:
        preprocessed_inputs["midi_duration_since_previous_onset"] = inputs["midi_duration_since_previous_onset"][...,None]
    preprocessed_inputs = {**inputs,**preprocessed_inputs}
    return preprocessed_inputs



     
