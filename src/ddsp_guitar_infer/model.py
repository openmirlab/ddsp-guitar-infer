"""model.py — control network + DDSP synthesis wrapper (★ load-bearing).

Inference-only replica of the Lightning module used at training time:
embeds string index and discrete/continuous MIDI features, runs a stack
of self-attention+RNN blocks (SARNNBlock), decodes harmonic/noise/f0
parameters, and drives ddsp_guitar_utils.synthesis_model.DDSPModel to
render audio from those parameters.

Reads: ddsp_guitar_utils.synthesis_model.DDSPModel ·
ddsp_guitar_utils.control_blocks.SARNNBlock · ddsp_guitar_utils.dsp.Quantizer
"""

from __future__ import annotations

from typing import Dict, List

import einops
import torch

from ddsp_guitar_infer.ddsp_guitar_utils import synthesis_model
from ddsp_guitar_infer.ddsp_guitar_utils.control_blocks import SARNNBlock
from ddsp_guitar_infer.ddsp_guitar_utils.dsp import Quantizer

GUITAR_F0_MIN_HZ = 35.0
GUITAR_F0_MAX_HZ = 1200.0


class GuitarControlModel(torch.nn.Module):
    """Lightweight replica of the Lightning module used for inference."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.quantizers: Dict[str, Quantizer] = {}
        self._init_quantizers()
        self._init_embeddings()
        self._init_main_block()
        self._init_output_block()
        self._init_ddsp_model()

    # ------------------------------------------------------------------
    # Initialization helpers
    def _init_quantizers(self) -> None:
        for feature in self.config.get("classification_features", []):
            value_range = tuple(feature["range"])
            self.quantizers[feature["name"]] = Quantizer(value_range, feature["n_bins"])

    def _init_embeddings(self) -> None:
        self.string_embedding_layer = torch.nn.Embedding(
            self.config["n_voices"], self.config["hidden_size"]
        )

        self.input_embedding_layers = torch.nn.ModuleDict()
        self.input_quantizers = {}
        for feature in self.config.get("discrete_inputs", []):
            feature_name = feature["name"]
            self.input_quantizers[feature_name] = Quantizer(tuple(feature["range"]), feature["n_bins"])
            self.input_embedding_layers[feature_name] = torch.nn.Embedding(
                feature["n_bins"], self.config["hidden_size"]
            )

        continuous_inputs = self.config.get("continuous_inputs", [])
        if continuous_inputs:
            input_size = self.config["hidden_size"] + len(continuous_inputs)
            self.input_block = torch.nn.Sequential(
                torch.nn.Linear(input_size, self.config["hidden_size"]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]),
                torch.nn.ReLU(),
            )
        else:
            self.input_block = None

    def _init_main_block(self) -> None:
        architecture = self.config.get("architecture", "lstm")
        if architecture not in {"lstm", "gru", "rnn"}:
            raise ValueError(f"Unsupported architecture for inference: {architecture}")

        blocks: List[torch.nn.Module] = []
        for _ in range(self.config["n_blocks"]):
            blocks.append(
                SARNNBlock(
                    input_channels=self.config["n_voices"],
                    num_heads=self.config.get("n_heads", 4),
                    hidden_size=self.config["hidden_size"],
                    rnn_type=architecture,
                    n_rnn_layers_per_block=self.config.get("n_rnn_layers_per_block", 1),
                    norm_type=self.config.get("norm_type", "layer norm c t f"),
                )
            )
        self.main_block = torch.nn.Sequential(*blocks)

    def _init_output_block(self) -> None:
        total_class = sum(f["n_bins"] for f in self.config.get("classification_features", []))
        total_reg = sum(f["n_features"] for f in self.config.get("regression_features", []))
        self.total_classification_output_size = total_class
        self.total_regression_output_size = total_reg
        self.output_block = torch.nn.Sequential(
            torch.nn.Linear(self.config["hidden_size"], total_class + total_reg)
        )

    def _init_ddsp_model(self) -> None:
        N_HARMONICS = self.config.get("n_harmonics", 128)
        N_NOISE_BANDS = self.config.get("n_noise_bands", 128)
        ir_duration = self.config.get("ir_duration", 0.25)

        self.ddsp_model = synthesis_model.DDSPModel(
            sample_rate=self.config["model_sample_rate"],
            n_harmonics=N_HARMONICS,
            n_noise_bands=N_NOISE_BANDS,
            ir_duration=ir_duration,
            use_one_ir_per_voice=True,
            input_ft_splits=(),
            get_decoder=lambda *_: None,
            min_f0_hz=GUITAR_F0_MIN_HZ,
            max_f0_hz=GUITAR_F0_MAX_HZ,
            voice_embedding_size=self.config["hidden_size"],
            noise_bias=self.config.get("noise_bias", -3.0),
            n_voices=self.config["n_voices"],
        )

    # ------------------------------------------------------------------
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, channel, time, _ = inputs["midi_pitch_scaled"].shape
        string_index = torch.arange(0, channel, device=inputs["midi_pitch_scaled"].device)
        string_z = self.string_embedding_layer(string_index)
        input_z = einops.repeat(string_z, "c f -> b c t f", b=batch, c=channel, t=time)

        for feature in self.config.get("discrete_inputs", []):
            name = feature["name"]
            values = inputs[name]
            values = self.input_quantizers[name].quantize(values)
            feature_z = self.input_embedding_layers[name](values)
            input_z = input_z + feature_z

        for feat in self.config.get("continuous_inputs", []):
            input_z = torch.cat([input_z, inputs[feat["name"]]], dim=-1)

        if self.input_block is not None:
            input_z = self.input_block(input_z)

        return input_z

    def decode(self, out: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        out = self.output_block(out)
        class_split = torch.split(
            out[..., : self.total_classification_output_size],
            [f["n_bins"] for f in self.config.get("classification_features", [])],
            dim=-1,
        )
        reg_start = self.total_classification_output_size
        reg_tensors = torch.split(
            out[..., reg_start:],
            [f["n_features"] for f in self.config.get("regression_features", [])],
            dim=-1,
        )

        for tensor, feature in zip(class_split, self.config.get("classification_features", [])):
            name = feature["name"]
            result[f"{name}_logits"] = tensor
            pred_class = torch.argmax(tensor, dim=-1)
            result[f"{name}_hat"] = self.quantizers[name].dequantize(pred_class)

        for tensor, feature in zip(reg_tensors, self.config.get("regression_features", [])):
            name = feature["name"]
            if name == "harmonic_partial_amp_output":
                result[name] = tensor
            elif name == "harmonic_global_amp_output":
                result[name] = tensor
            elif name == "noise_band_amp_output":
                result[name] = tensor
            elif name == "hex_f0_scaled":
                result[f"{name}_hat"] = torch.sigmoid(tensor)
            else:
                result[f"{name}_hat"] = tensor

        return result

    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self.encode(inputs)
        hidden = self.main_block(encoded)
        decoded = self.decode(hidden, inputs)
        decoded["voice_index"] = inputs.get("voice_index")
        return decoded

    def render(self, predictions: Dict[str, torch.Tensor], n_samples: int | None = None) -> Dict[str, torch.Tensor]:
        synth_inputs = {
            "harmonic_partial_amp_output": predictions["harmonic_partial_amp_output"],
            "harmonic_global_amp_output": predictions["harmonic_global_amp_output"],
            "noise_band_amp_output": predictions["noise_band_amp_output"],
            "voice_index": predictions["voice_index"],
            "hex_f0_scaled": predictions["hex_f0_scaled_hat"],
        }
        synth_device = next(self.ddsp_model.parameters()).device
        synth_inputs = {
            key: value.to(synth_device) if isinstance(value, torch.Tensor) else value
            for key, value in synth_inputs.items()
        }
        return self.ddsp_model.synthesize(
            synth_inputs,
            self.config["n_samples"] if n_samples is None else n_samples,
        )


__all__ = ["GuitarControlModel", "GUITAR_F0_MIN_HZ", "GUITAR_F0_MAX_HZ"]
