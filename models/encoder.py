import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, prenorm=False, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.prenorm = prenorm
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.prenorm:
            new_x, attn= self._sa_block(self.norm1(x),attn_mask)
            x = x + new_x
            x = x + self._ff_block(self.norm2(x))
        else:
            new_x, attn = self._sa_block(x, attn_mask)
            x = self.norm1(x + new_x)
            x = self.norm2(x + self._ff_block(x))
        return x, attn

    def _sa_block(self, x, attn_mask):
        x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
            )
        return self.dropout(x), attn

    def _ff_block(self, x):
        x = self.conv2(self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))).transpose(-1, 1)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns,


