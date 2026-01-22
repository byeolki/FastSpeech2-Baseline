import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FFTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        kernel_size: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(
            d_model, d_ffn, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            d_ffn, d_model, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, mask)
        x = self.dropout1(x)
        x = self.norm1(residual + x)

        residual = x
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.dropout2(x)
        x = self.norm2(residual + x)

        return x
