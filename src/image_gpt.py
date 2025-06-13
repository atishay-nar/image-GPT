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
        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        # transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            activation= "gelu", # prev relu
            dim_feedforward = 4 * embed_dim,
            batch_first = True
        )

        self.transformer = nn.TransformerDecoder(transformer_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab, bias=False)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device

        # embed
        h = self.embed_tokens(x)

        # prepend sos token
        sos = torch.ones(1, batch_size, len(h), device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], dim=0)

        # embed tokens and postions
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # (B, S)
        h =  h + self.pos_embed(positions)  # (B, S, embed_dim)

        # create mask
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float("-inf"), diagonal=1) # (S, S)

        out = self.transformer(
            tgt=h,
            memory=h,
            tgt_mask=mask
        )

        # final layer norm and head
        out = self.ln_f(out)
        logits = self.head(out)
        return logits
