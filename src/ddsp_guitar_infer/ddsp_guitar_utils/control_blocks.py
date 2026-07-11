"""Control-network blocks used by the guitar inference model."""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
from torch import nn


class SARNNBlock(nn.Module):
    """Self-attention + bidirectional RNN block with residual connections."""

    def __init__(
        self,
        input_channels: int,
        num_heads: int,
        hidden_size: int,
        rnn_type: str,
        n_rnn_layers_per_block: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = n_rnn_layers_per_block
        self.norm_type = norm_type

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.rnn = self._create_rnn(rnn_type, hidden_size, n_rnn_layers_per_block)

    def _create_rnn(self, rnn_type: str, hidden_size: int, num_layers: int) -> nn.Module:
        rnn_type = rnn_type.lower()
        if rnn_type == "rnn":
            return nn.RNN(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        if rnn_type == "lstm":
            return nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        if rnn_type == "gru":
            return nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, time, feat = x.shape
        assert channel == self.input_channels, "channel mismatch"
        if self.norm_type == "layer norm c t f":
            x = F.layer_norm(x, x.shape[1:])

        attn_in = einops.rearrange(x, "b c t f -> (b t) c f")
        attn_out = self.self_attn(attn_in, attn_in, attn_in)[0]
        attn_out = einops.rearrange(attn_out, "(b t) c f -> b c t f", b=batch, c=channel, t=time)
        x = x + attn_out

        rnn_in = einops.rearrange(x, "b c t f -> (b c) t f")
        rnn_out, _ = self.rnn(rnn_in)
        hidden = rnn_out[:, :, :self.hidden_size] + rnn_out[:, :, self.hidden_size:]
        hidden = einops.rearrange(hidden, "(b c) t f -> b c t f", b=batch, c=channel, t=time)
        return x + hidden


__all__ = ["SARNNBlock"]
