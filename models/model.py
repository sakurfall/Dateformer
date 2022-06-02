import torch
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.decoder import Decoder, DecoderLayer
from models.dert import DertEncoder
from models.embed import DateEmbedding
from models.encoder import Encoder, EncoderLayer
from models.embed import PositionalEmbedding


class Dateformer(nn.Module):
    def __init__(self, *, d_features, d_sequences, d_model, d_ff=1024, e_layers, d_layers,n_heads=8,
                 granularity, n_predays, n_postdays, max_len, prenorm=False, time_mapper=None,
                 attn='full', factor = 5, activation='gelu', dropout=0.1, output_attention=False,
                 device=torch.device('cuda:0')):
        super(Dateformer, self).__init__()

        self.encoder = DertEncoder(d_features=d_features,d_model=d_model,e_layers=e_layers[0],n_heads=n_heads,d_ff=d_ff,index=n_predays,
                                   attn=attn,factor=factor,activation=activation,prenorm=prenorm,dropout=dropout,
                                   output_attention=output_attention,device=device)

        self.timeEmbedding = PositionalEmbedding(d_model=d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU() if activation=='relu' else nn.GELU(),
            nn.Linear(d_ff,d_sequences)
        )

        if time_mapper == 'share':
            predictor = self.decoder
        elif time_mapper == 'customize':
            predictor = nn.Sequential(
                nn.Linear(d_model,d_ff),
                nn.ReLU() if activation=='relu' else nn.GELU(),
                nn.Linear(d_ff,d_sequences)
            )
        else: predictor = None

        self.residualLearner = Longlongformer(d_features=d_model,d_sequences=d_sequences,d_model=d_model,e_layers=e_layers[1],d_layers=d_layers,n_heads=n_heads,d_ff=d_ff,
                                               attn=attn,factor=factor,granularity=granularity,activation=activation,output_attention=output_attention,
                                               prenorm=prenorm,dropout=dropout,device=device,predictor=predictor)

        self.activation = F.relu if activation == "relu" else F.gelu

        self.granularity = granularity

        self.n_predays = n_predays
        self.n_postdays = n_postdays
        self.output_attention = output_attention
        self.max_len = max_len

    def forward(self, x_date, x_lookback, lookback_window=None):

        vecs = [];encoder_attns = [];residual_attns = []
        for i in range(self.n_predays + self.n_postdays + 1, x_date.shape[1] + 1):
            vec, attns = self.encoder(x_date[:,i-self.n_predays-self.n_postdays-1:i,:])
            vecs.append(vec);encoder_attns.append(attns)

        vecs = torch.stack(vecs,dim=1)
        dec_in = repeat(vecs,'b d w->b d h w',h=self.granularity)
        dec_in = dec_in + self.timeEmbedding(dec_in)

        dec_out = self.decoder(dec_in)

        n_lookbackdays = x_lookback.shape[1]
        n_predictdays = vecs.shape[1] - n_lookbackdays
        residuals = x_lookback - dec_out[:,:n_lookbackdays,...]

        # learn localized prediction through Autoregressive day-by-day
        for i in range(x_lookback.shape[1], vecs.shape[1]):
            if lookback_window:
                j = i - round(lookback_window)
                lookback_window += 1
                if lookback_window < 15: lookback_window += 1
                elif lookback_window < 30: lookback_window += 0.74
                elif lookback_window < 90: lookback_window += 0.49
                elif lookback_window < 180: lookback_window += 0.24
                else: lookback_window += 0.1
                if lookback_window>self.max_len-1:
                    lookback_window = self.max_len-1
            else:j = 0

            if j < 0: j = 0
            residual, residual_attn = self.residualLearner(vecs[:,j:i+1,...],residuals[:,j:i,...])
            residuals = torch.cat([residuals,torch.unsqueeze(residual,dim=1)],dim=1)
            residual_attns.append(residual_attn)

        assert residuals.shape == dec_out.shape

        out = dec_out + residuals

        if self.output_attention:
            return out[:, -n_predictdays:, ...], encoder_attns, residual_attns
        else:
            return out[:, -n_predictdays:, ...]


class Longlongformer(nn.Module):

    def __init__(self, *, d_features, d_sequences, d_model, d_ff=1024, e_layers, d_layers=1, n_heads=8, predictor=None, granularity=96, attn='full', factor = 5,
                 activation='gelu', prenorm=False, dropout=0.1,output_attention = False,mix = True,
                 device=torch.device('cuda:0')):
        super(Longlongformer, self).__init__()

        # to patch
        self.enc_embedding = nn.Sequential(
            Rearrange('b d h w->b d (h w)'),
            DateEmbedding(granularity*d_sequences, d_model, dropout)
        )

        self.dec_embedding = DateEmbedding(d_features, d_model, dropout)

        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
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

        self.enc_to_dec = nn.Identity()

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    d_model,
                    d_ff,
                    prenorm=prenorm,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)

        self.timeEmbedding = PositionalEmbedding(d_model=d_model)

        self.linear3 = nn.Linear(d_ff, d_sequences)
        self.predict = predictor

        self.activation = F.relu if activation == "relu" else F.gelu
        self.granularity = granularity
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_date, residuals):

        assert x_date.shape[1] - 1 == residuals.shape[1]

        enc_in = self.enc_embedding(residuals)
        enc_out, attns = self.encoder(enc_in)

        cross = self.enc_to_dec(enc_out)

        dec_in = self.dec_embedding(x_date)
        dec_out = self.decoder(dec_in, cross)

        if self.predict:
            logits = self.dropout(self.linear3(self.dropout(self.activation(self.linear1(dec_out)))))
            embedding = dec_out[:, -1, :]
            embedding = embedding + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(embedding)))))
            embedding = repeat(embedding, 'b w->b h w', h=self.granularity)
            embedding = embedding + self.timeEmbedding(embedding)
            prediction = self.predict(embedding).unsqueeze(dim=1)
            residuals = torch.cat((residuals, prediction), dim=1)
        else:
            logits = self.dropout(self.linear3(self.dropout(self.activation(self.linear1(dec_out[:, :-1, :])))))

        distribution = torch.softmax(logits, dim=1)
        out = torch.einsum('b d h w,b d w->b h w', residuals, distribution)
        return out, attns
