import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_layers, n_heads, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_dim, d_model, n_layers, n_heads, dropout)
        self.decoder = Decoder(output_dim, d_model, n_layers, n_heads, dropout)
        self.linear = nn.Linear(d_model, output_dim)

    def forward(self, src, trg):
        enc_output = self.encoder(src)
        dec_output = self.decoder(trg, enc_output)
        output = self.linear(dec_output)
        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, src):
#         x = self.embedding(src.float()) 
        x = self.embedding(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Linear(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, trg, enc_output):
        x = self.embedding(trg)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.layer_norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.layer_norm2(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_output):
        trg2 = self.self_attn(trg, trg, trg)
        trg = trg + self.dropout(trg2)
        trg = self.layer_norm1(trg)
        trg2 = self.enc_dec_attn(trg, enc_output, enc_output)
        trg = trg + self.dropout(trg2)
        trg = self.layer_norm2(trg)
        trg2 = self.feed_forward(trg)
        trg = trg + self.dropout(trg2)
        trg = self.layer_norm3(trg)
        return trg

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        def split_heads(x, batch_size):
            x = x.view(batch_size, -1, self.n_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)

        Q = split_heads(Q, batch_size)
        K = split_heads(K, batch_size)
        V = split_heads(V, batch_size)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention = F.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)