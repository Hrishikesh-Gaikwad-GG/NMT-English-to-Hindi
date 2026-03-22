import torch
import torch.nn as nn
import math

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
    

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        emb_dim=256,
        n_heads=8,
        n_layers=3,
        dropout=0.2
    ):
        super().__init__()
        
        self.src_embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.trg_embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(emb_dim)
        
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.emb_dim = emb_dim
        
    def forward(self, src, trg):
        
        src_mask = (src == 0)
        trg_mask = (trg == 0)
        
        tgt_seq_len = trg.size(1)


        causal_mask = torch.triu(
            torch.full((tgt_seq_len, tgt_seq_len), float("-inf")),
            diagonal=1
        ).to(src.device)
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.emb_dim))
        trg_emb = self.pos_encoder(self.trg_embedding(trg) * math.sqrt(self.emb_dim))
        
        output = self.transformer(
            src_emb,
            trg_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=trg_mask,
            memory_key_padding_mask=src_mask,
            tgt_mask=causal_mask
            )
        
        return self.fc_out(output)