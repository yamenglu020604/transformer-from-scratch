import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # For head dimension
        
        batch_size = query.size(0)

        # 1) Linear projections -> (batch_size, n_heads, seq_len, d_k)
        query, key, value = [
            l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Scaled Dot-Product Attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # 3) Apply attention to value
        x = torch.matmul(p_attn, value)

        # 4) Concat and final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # # Sublayer 1: Multi-head attention
        # src2 = self.self_attn(src, src, src, src_mask)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)

        # # Sublayer 2: Feed-forward network
        # src2 = self.feed_forward(src)
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)

        #消融实验：验证残差连接和层归一化
        src = self.self_attn(src, src, src, src_mask)
        src = self.feed_forward(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        # # Sublayer 1: Masked multi-head self-attention
        # tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        # # Sublayer 2: Multi-head cross-attention
        # tgt2 = self.cross_attn(tgt, memory, memory, src_mask)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # # Sublayer 3: Feed-forward network
        # tgt2 = self.feed_forward(tgt)
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)


        # 消融实验：移除残差连接与层归一化
        tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.cross_attn(tgt, memory, memory, src_mask)
        tgt = self.feed_forward(tgt)
        return tgt

class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.linears[0].in_features)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

class Decoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.linears[0].in_features)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return self.norm(tgt)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, n_layers)

        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, n_layers)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        # src_mask: [batch_size, 1, src_len]
        # tgt_mask: [batch_size, tgt_len, tgt_len]
        
        # 置换维度[seq_len, batch_size, d_model]
        src_emb = self.pos_encoder(self.encoder_embedding(src).permute(1, 0, 2))
        tgt_emb = self.pos_encoder(self.decoder_embedding(tgt).permute(1, 0, 2))

        # # 移除位置编码,用于消融实验
        # src_emb = self.encoder_embedding(src).permute(1, 0, 2)
        # tgt_emb = self.decoder_embedding(tgt).permute(1, 0, 2)

        #置换维度
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        
        return self.fc_out(output)

    @staticmethod
    def create_masks(src, tgt, pad_idx, device):
        src_mask = (src != pad_idx).unsqueeze(1) # (batch, 1, src_len)
        
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1) # (batch, 1, tgt_len)
        tgt_len = tgt.size(1)
        tgt_look_ahead_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool() # (tgt_len, tgt_len)
        
        tgt_mask = tgt_pad_mask & tgt_look_ahead_mask # (batch, tgt_len, tgt_len)
        
        return src_mask, tgt_mask