import torch
import torch.nn as nn

from ..common import FFTBlock, PositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ffn: int,
        n_mel_channels: int,
        kernel_size: int = 9,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                FFTBlock(d_model, n_heads, d_ffn, kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

        self.mel_linear = nn.Linear(d_model, n_mel_channels)

        self.postnet = nn.Sequential(
            nn.Conv1d(n_mel_channels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, n_mel_channels, kernel_size=5, padding=2),
            nn.Dropout(0.5),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        mel_out = self.mel_linear(x)

        mel_postnet = self.postnet(mel_out.transpose(1, 2))
        mel_postnet = mel_postnet.transpose(1, 2)
        mel_postnet = mel_out + mel_postnet

        return mel_out, mel_postnet
