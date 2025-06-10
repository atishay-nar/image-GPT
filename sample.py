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

from test import TransformerGPT


# ───────────────────────────────────────────────────────────────────────────────
# Hyperparameters (must match training)
# ───────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 28               # model resolution: 28x28
SEQ_LEN      = IMAGE_SIZE * IMAGE_SIZE  # 794
NUM_CLUSTERS = 32              # vocabulary size used at training
EMBED_DIM    = 32
NUM_HEADS    = 2
NUM_LAYERS   = 8

DEVICE = (
    "mps" if torch.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)



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
    img = img_pil.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))#, Image.BILINEAR) # make single channel


    arr = np.array(img, dtype=np.uint8).reshape(-1, 1)                       # flatten to shape: (784,1)

    # ensure correct type
    pix = arr.astype(np.int32)                                # (784,1)
    cents = centroids.astype(np.int32)                        # (28,1)

    pix_norm = np.sum(pix * pix, axis=1, keepdims=True)       # (784,1)
    cent_norm = np.sum(cents * cents, axis=1)                 # (28,)
    dot = pix @ cents.T                                       # (784,28)
    dists = pix_norm + cent_norm - 2 * dot                     # (784,28)

    token_ids = np.argmin(dists, axis=1)                       # (784,)
    return torch.from_numpy(token_ids).long()                  # (784,)


# ───────────────────────────────────────────────────────────────────────────────
# Autoregressively sample missing tokens given a prefix (first 392 tokens)
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def conditional_sample(model: TransformerGPT,
                       prefix_tokens: torch.Tensor,
                       total_length: int,
                       temperature: float = 1.0,
                       top_k: int = 50) -> torch.Tensor:
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
def tokens_to_image(token_seq: torch.Tensor, centroids: np.ndarray):
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
        "--top_k", type=int, default=8,
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
