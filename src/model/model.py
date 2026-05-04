import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Adds positional information to the sequence so the Transformer knows the order of frames."""

    def __init__(self, d_model: int, max_len: int = 150, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):
    """Projects keypoint features to model dimension, adds positional encoding, and runs Transformer encoder layers."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, nhead: int = 8):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.pos_encoding(x)
        return self.transformer(x)


class Decoder(nn.Module):
    """Generates output tokens autoregressively using the encoder output as context."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, vocab_size: int = 30522, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.transformer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output_projection(x)
 
    
class SignLanguageTransformer(nn.Module):
    """Full seq2seq model that encodes keypoint sequences and decodes them into text tokens."""

    def __init__(self, config: dict):
        super().__init__()
        m = config['model']
        self.encoder = Encoder(
            input_dim=m['input_dim'],
            hidden_dim=m['encoder_hidden_dim'],
            num_layers=m['encoder_layers'],
            dropout=m['encoder_dropout'],
        )
        self.decoder = Decoder(
            hidden_dim=m['decoder_hidden_dim'],
            num_layers=m['decoder_layers'],
            dropout=m['decoder_dropout'],
        )

    def make_causal_mask(self, size: int) -> torch.Tensor:
        """Generates a causal mask to prevent the decoder from attending to future tokens."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        memory = self.encoder(src)
        tgt_mask = self.make_causal_mask(tgt.size(1)).to(src.device)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)    