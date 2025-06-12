#imports
import torch
import torch.nn as nn


# set device
DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
        )

# decode-only Transfromer model for image generation
class ImageGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vocab = cfg.NUM_CLUSTERS
        embed_dim = cfg.EMBED_DIM
        num_heads = cfg.NUM_HEADS
        num_layers = cfg.NUM_LAYERS
        max_seq_len = cfg.IMAGE_SIZE * cfg.IMAGE_SIZE

        # embeds vocab (centroids) into model dimension
        self.embed_tokens = nn.Embedding(vocab, embed_dim)
        # positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            activation= "gelu", # prev relu
            dim_feedforward = 4 * embed_dim,
            dropout = 0.0, # prev 0.1
            batch_first = True
        )

        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab, bias=False)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # embed tokens and postions
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # (B, S)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)  # (B, S, embed_dim)

        # create mask
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float("-inf"), diagonal=1) # (S, S)

        out = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=mask
        )

        # final layer norm and head
        out = self.ln_f(out)
        logits = self.head(out)
        return logits
