
import torch
from einops import repeat
from torch import nn

import torch.nn.functional as F
from models.attn import ProbAttention, FullAttention, AttentionLayer
from models.embed import DateEmbedding, PositionalEmbedding
from models.encoder import Encoder, EncoderLayer


class DertEncoder(nn.Module):
    def __init__(self, *, d_features, d_model, d_ff=512, e_layers, n_heads=8, index, attn='full', factor = 5,
                 activation='gelu', prenorm=False, dropout=0.1, output_attention=False,
                 device=torch.device('cuda:0')):
        super(DertEncoder, self).__init__()

        self.enc_embedding = DateEmbedding(d_features, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    prenorm=prenorm,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.index = index

    def forward(self, x):

        enc_in = self.enc_embedding(x)
        enc_out, attn = self.encoder(enc_in)

        representation = enc_out[:, self.index, :]

        return representation, attn


class Dert(nn.Module):
    def __init__(self, *, d_features, d_sequences, d_model, e_layers, n_heads=8, d_ff=512, n_predays,
                  attn='full', factor=5, granularity=96,
                 activation='gelu', output_attention=False, prenorm=False, dropout=0.1,
                 device=torch.device('cuda:0')):
        super(Dert, self).__init__()
        self.encoder = DertEncoder(d_features=d_features, d_model=d_model, e_layers=e_layers, n_heads=n_heads, d_ff=d_ff,index=n_predays,
                                   attn=attn, factor=factor, activation=activation, prenorm=prenorm, dropout=dropout,
                                   output_attention=output_attention, device=device)

        self.timeEmbedding = PositionalEmbedding(d_model=d_model)
        self.granularity = granularity
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_ff, d_sequences)
        )
        self.output_attention = output_attention

    def forward(self, x_date):
        vec, attns = self.encoder(x_date)
        dec_in = repeat(vec, 'b w->b h w', h=self.granularity)
        dec_in = dec_in + self.timeEmbedding(dec_in)

        dec_out = self.decoder(dec_in)
        if self.output_attention:
            return dec_out, attns
        else: return dec_out


class DertforPretrain(nn.Module):
    def __init__(self, *, d_features, d_sequences, d_model, e_layers, n_heads=8, d_ff=512, n_predays,
                  attn='full', factor=5,
                 activation='gelu', output_attention=False, prenorm=False, dropout=0.1,
                 device=torch.device('cuda:0')):
        super(DertforPretrain, self).__init__()
        self.encoder = DertEncoder(d_features=d_features, d_model=d_model, e_layers=e_layers, n_heads=n_heads, d_ff=d_ff,index=n_predays,
                                   attn=attn, factor=factor, activation=activation, prenorm=prenorm, dropout=dropout,
                                   output_attention=output_attention, device=device)

        self.meanPredictHead = nn.Linear(d_model, d_sequences)
        self.correlationPredictHead = nn.Linear(d_model, d_sequences)
        self.norm = nn.BatchNorm1d(d_sequences)

    def forward(self, x_date, y_sequences):
        vec, attns = self.encoder(x_date)
        mean_pred = self.meanPredictHead(vec)
        correlation_pred = self.correlationPredictHead(vec)

        mean = torch.mean(y_sequences, dim=-2)
        assert mean.size() == mean_pred.size()
        mean_loss = F.mse_loss(mean_pred, mean)

        q_fft = torch.fft.rfft(y_sequences, dim=-2)
        norm = torch.linalg.norm(y_sequences, dim=-2)
        norm = torch.unsqueeze(norm * norm, dim=-2)
        correlation = torch.fft.irfft(q_fft*q_fft.conj(), dim=1)/norm
        correlation = torch.mean(correlation,dim=-2)
        correlation = self.norm(correlation)
        assert correlation.size() == correlation_pred.size()
        correlation_loss = F.mse_loss(correlation_pred, correlation)
        return mean_loss + correlation_loss


class DertWithCorrelateHeadforPretrain(nn.Module):
    def __init__(self, *, d_features, d_sequences, d_model, e_layers, n_heads=8, d_ff=512, n_predays,
                  attn='full', factor=5,
                 activation='gelu', output_attention=False, prenorm=False, dropout=0.1,
                 device=torch.device('cuda:0')):
        super(DertWithCorrelateHeadforPretrain, self).__init__()
        self.encoder = DertEncoder(d_features=d_features, d_model=d_model, e_layers=e_layers, n_heads=n_heads, d_ff=d_ff,index=n_predays,
                                   attn=attn, factor=factor, activation=activation, prenorm=prenorm, dropout=dropout,
                                   output_attention=output_attention, device=device)

        self.correlationPredictHead = nn.Linear(d_model, d_sequences)
        self.norm = nn.BatchNorm1d(d_sequences)

    def forward(self, x_date, y_sequences):
        vec, attns = self.encoder(x_date)

        correlation_pred = self.correlationPredictHead(vec)

        q_fft = torch.fft.rfft(y_sequences, dim=-2)
        correlation = torch.fft.irfft(q_fft * q_fft.conj(), dim=1)
        correlation = torch.mean(correlation, dim=-2)
        correlation = self.norm(correlation)
        assert correlation.size() == correlation_pred.size()
        correlation_loss = F.mse_loss(correlation_pred, correlation)
        return correlation_loss


class DertWithMeanHeadforPretrain(nn.Module):
    def __init__(self, *, d_features, d_sequences, d_model, e_layers, n_heads=8, d_ff=512, n_predays,
                  attn='full', factor=5,
                 activation='gelu', output_attention=False, prenorm=False, dropout=0.1,
                 device=torch.device('cuda:0')):
        super(DertWithMeanHeadforPretrain, self).__init__()
        self.encoder = DertEncoder(d_features=d_features, d_model=d_model, e_layers=e_layers, n_heads=n_heads, d_ff=d_ff,index=n_predays,
                                   attn=attn, factor=factor, activation=activation, prenorm=prenorm, dropout=dropout,
                                   output_attention=output_attention, device=device)

        self.meanPredictHead = nn.Linear(d_model, d_sequences)

    def forward(self, x_date, y_sequences):
        vec, attns = self.encoder(x_date)
        mean_pred = self.meanPredictHead(vec)

        mean = torch.mean(y_sequences, dim=-2)
        assert mean.size() == mean_pred.size()
        mean_loss = F.mse_loss(mean_pred, mean)

        return mean_loss

