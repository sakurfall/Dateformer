import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,prenorm=False,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.prenorm = prenorm
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        if self.prenorm:
            x = x + self._sa_block(self.norm1(x), x_mask)[0]
            x = x + self._ca_block(self.norm2(x), cross, cross_mask)[0]
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, x_mask)[0])
            x = self.norm2(x + self._ca_block(x, cross, cross_mask)[0])
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x, attn = self.self_attention(
            x, x, x,
            attn_mask=attn_mask
            )
        return self.dropout(x), attn

    def _ca_block(self, x, cross, cross_mask):
        x, attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
            )
        return self.dropout(x), attn

    def _ff_block(self, x):
        x = self.conv2(self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))).transpose(-1, 1)
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
