#imports
import torch
import torch.nn as nn
import argparse
import yaml



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
        self.embed_dim = cfg.EMBED_DIM
        num_heads = cfg.NUM_HEADS
        num_layers = cfg.NUM_LAYERS
        max_seq_len = cfg.IMAGE_SIZE * cfg.IMAGE_SIZE

        # embeds vocab (centroids) into model dimension
        self.embed_tokens = nn.Embedding(vocab, self.embed_dim)
        # positional embedding
        self.pos_embed = nn.Embedding(max_seq_len, self.embed_dim)
        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(self.embed_dim))
        nn.init.normal_(self.sos)

        # transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model = self.embed_dim,
            nhead = num_heads,
            activation= "gelu", # prev relu
            dim_feedforward = 4 * self.embed_dim,
            batch_first = True
            
        )
        self.transformer = nn.ModuleList([transformer_layer for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab, bias=False)
    
    def forward(self, x):
        # x: (B, S) where B is batch size and S is sequence length
        batch_size, seq_len = x.size()
        device = x.device

        # embed each part of sequence with vector of size embed_dim
        h = self.embed_tokens(x)

        # prepend sos token
        sos = torch.ones(batch_size, 1, self.embed_dim, device=device) * self.sos
        h = torch.cat([sos, h[:, :-1, :]], dim=1)

        # embed tokens and postions
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (B, S)
        h =  h + self.pos_embed(positions)  # (B, S, embed_dim)

        # create mask
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float("-inf"), diagonal=1) # (S, S)

        # transformer layers
        for layer in self.transformer:
            h = layer(tgt=h, memory=h, tgt_mask=mask)

        # final layer norm and head
        out = self.ln_f(h)
        logits = self.head(out)
        return logits

if __name__ == "__main__":
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))
    model = ImageGPT(cfg)
    dummy = torch.ones((3, 16), dtype=torch.long)
    logits = model(dummy)