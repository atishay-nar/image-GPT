"""
sample.py

Load a MNIST test image, take its top half as context, and use a trained
TransformerGPT (28x28) to complete the bottom half. Displays original vs.
completed 28x28 images side by side so you can verify the model is working.

Usage:
    python sample.py \
        --checkpoint_path ./checkpoints/image_gpt_epoch{epoch}.pth \
        --centroids_path ./centroids_{num_clusters}.npy \
        [--index 0] [--top_k 50] [--temperature 1.0]

Arguments:
    --checkpoint_path  Path to your trained TransformerGPT .pth file.
    --centroids_path   Path to centroids_{num_clusters}.npy
    --index            (Optional) MNIST test image index (default: 0).
    --top_k            (Optional) Top-k sampling (default: 50).
    --temperature      (Optional) Sampling temperature (default: 1.0).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt


# ───────────────────────────────────────────────────────────────────────────────
# Hyperparameters (must match training)
# ───────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 28               # model resolution: 28x28
SEQ_LEN      = IMAGE_SIZE * IMAGE_SIZE  # 256
NUM_CLUSTERS = 16              # vocabulary size used at training
EMBED_DIM    = 16
NUM_HEADS    = 2
NUM_LAYERS   = 6

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ───────────────────────────────────────────────────────────────────────────────
# TransformerGPT definition (must match exactly your training script)
# ───────────────────────────────────────────────────────────────────────────────
class TransformerGPT(nn.Module):
    def __init__(self,
                 vocab_size: int = NUM_CLUSTERS,
                 embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS,
                 max_seq_len: int = SEQ_LEN):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed    = nn.Embedding(max_seq_len, embed_dim)

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            transformer_layer,
            num_layers=num_layers
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def _generate_causal_mask(self, L: int, device: torch.device):
        return torch.triu(torch.ones((L, L), device=device) * float("-inf"), diagonal=1)

    def forward(self, input_ids: torch.LongTensor):
        B, S = input_ids.size()
        device = input_ids.device
        positions = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)
        causal_mask = self._generate_causal_mask(S, device=device)
        out = self.transformer(tgt=x, memory=x, tgt_mask=causal_mask)
        out = self.ln_f(out)
        logits = self.head(out)
        return logits  # shape: (B, S, vocab_size)


# ───────────────────────────────────────────────────────────────────────────────
# Quantize a 28x28 PIL image into a 784-length token array via centroids
# ───────────────────────────────────────────────────────────────────────────────
def quantize_image_to_tokens(img_pil: Image.Image, centroids: np.ndarray):
    """
    Args:
      img_pil: PIL.Image in RGB. Will be resized to 28x28
      centroids: NumPy array (16,1) of uint8 centroid colors.

    Returns:
      tokens: torch.LongTensor of shape (784,), each in [0,16).
    """
    img = img_pil.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR) # make single channel


    arr = np.array(img, dtype=np.uint8).reshape(-1, 1)                       # flatten to shape: (784,1)

    # ensure correct type
    pix = arr.astype(np.int32)                                # (784,1)
    cents = centroids.astype(np.int32)                        # (16,1)

    pix_norm = np.sum(pix * pix, axis=1, keepdims=True)       # (784,1)
    cent_norm = np.sum(cents * cents, axis=1)                 # (16,)
    dot = pix @ cents.T                                       # (784,16)
    dists = pix_norm + cent_norm - 2 * dot                     # (784,16)

    token_ids = np.argmin(dists, axis=1)                       # (784,)
    return torch.from_numpy(token_ids).long()                  # (784,)


# ───────────────────────────────────────────────────────────────────────────────
# Autoregressively sample missing tokens given a prefix (first 392 tokens)
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def conditional_sample(model: TransformerGPT,
                       prefix_tokens: torch.LongTensor,
                       total_length: int,
                       temperature: float = 1.0,
                       top_k: int = 50) -> torch.LongTensor:
    """
    Args:
      model: trained TransformerGPT.
      prefix_tokens: 1D LongTensor of length 392.
      total_length: 784.
      temperature: float.
      top_k: int.

    Returns:
      full_tokens: LongTensor shape (784,) containing prefix + generated tokens.
    """
    device = next(model.parameters()).device
    prefix_len = prefix_tokens.size(0)
    generated = prefix_tokens.unsqueeze(0).to(device)  # shape: (1,392)

    for pos in range(prefix_len, total_length):
        logits_all = model(generated)                    # (1, pos, 16)
        logits_last = logits_all[:, -1, :] / temperature  # (1,16)

        if top_k is not None:
            vals, _ = torch.topk(logits_last, top_k)
            min_val = vals[:, -1].unsqueeze(1)             # (1,1)
            logits_last = torch.where(
                logits_last < min_val,
                torch.full_like(logits_last, float("-inf")),
                logits_last
            )

        probs = torch.softmax(logits_last, dim=-1)        # (1,16)
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
        generated = torch.cat((generated, next_token), dim=1) # (1, pos+1)

    return generated.squeeze(0).cpu()  # shape: (256,)


# ───────────────────────────────────────────────────────────────────────────────
# Convert a full 784-length token sequence back to a 28x28 gray image
# ───────────────────────────────────────────────────────────────────────────────
def tokens_to_image(token_seq: torch.LongTensor, centroids: np.ndarray):
    """
    Args:
      token_seq: LongTensor (256,), each in [0,16).
      centroids: NumPy array (16,1) of uint8.

    Returns:
      PIL.Image (32×32) upsampled with nearest neighbor for display.
    """
    tokens_np = token_seq.numpy().astype(np.int32)  # (784)
    grays = centroids[tokens_np]                      # (784), uint8
    arr = grays.reshape((IMAGE_SIZE, IMAGE_SIZE))  # (28,28)
    img28 = Image.fromarray(arr, mode="L")
    return img28


# ───────────────────────────────────────────────────────────────────────────────
# Main: load MNIST test image, quantize prefix, sample, and display results
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to trained TransformerGPT .pth checkpoint"
    )
    parser.add_argument(
        "--centroids_path", type=str, required=True,
        help="Path to centroids_16.npy (NUM_CLUSTERS=16)"
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Index of MNIST test image to use (default: 0)"
    )
    parser.add_argument(
        "--top_k", type=int, default=16,
        help="Top-k sampling (default: 16)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    args = parser.parse_args()

    # 1) Load centroids
    if not os.path.exists(args.centroids_path):
        raise FileNotFoundError(f"Centroids file not found: {args.centroids_path}")
    centroids = np.load(args.centroids_path)  # shape (16, 1), dtype=uint8

    # 2) Load MNISt test dataset, pick one image
    test_set = datasets.MNIST(root="./data", train=False, download=True)
    img28, _ = test_set[args.index]  # PIL.Image of size 28*28


    # 3) Quantize full 28x28 image to tokens, then split prefix (first 128 tokens)
    full_tokens = quantize_image_to_tokens(img28, centroids)  # (28*28,)
    prefix_len = SEQ_LEN // 2  # 28*28 // 2 = 392
    prefix_tokens = full_tokens[:prefix_len]                  # (392,)

    # 4) Load model and checkpoint
    model = TransformerGPT(
        vocab_size=NUM_CLUSTERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=SEQ_LEN
    ).to(DEVICE)

    ckpt = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()

    # 5) Sample bottom half tokens conditioned on top half
    generated = conditional_sample(
        model,
        prefix_tokens=prefix_tokens,
        total_length=SEQ_LEN,
        temperature=args.temperature,
        top_k=args.top_k
    )  # (784,)

    # Overwrite first 392 tokens with the original prefix
    generated[:prefix_len] = prefix_tokens

    # 6) Reconstruct completed image
    completed_img = tokens_to_image(generated, centroids)  # PIL.Image 28*28

    # 7) Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img28)
    axes[0].set_title("Top Half (Given)")
    axes[0].axis("off")

    axes[1].imshow(completed_img)
    axes[1].set_title("Model Completion")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
