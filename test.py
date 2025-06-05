"""
image_gpt.py

A PyTorch implementation of Image GPT (iGPT-S style) for CIFAR-10, updated to work with
the latest versions of the required libraries:

- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- tqdm >= 4.64.0
- Pillow >= 9.5.0
- scikit-learn >= 1.2.0

This script reproduces the core functionality of the repository:
https://github.com/teddykoker/image-gpt

It performs:
1. Computing k-means centroids over MNIST pixels to quantize the intensity space.
2. A Dataset that yields quantized token sequences for each image.
3. A simple GPT-like (decoder-only) Transformer that autoregressively models pixel tokens.
4. A training loop for generative pretraining on MNIST.

Usage:
    python image_gpt.py --mode compute_centroids
    python image_gpt.py --mode train --epochs 50

Requirements:
    pip install torch torchvision numpy tqdm pillow scikit-learn
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from sklearn.cluster import MiniBatchKMeans

###############################################################################
#                             Configuration                                    #
###############################################################################

# Default directories
DATA_DIR = "./data"

TOKENIZED_DATA_DIR = "./tokenized_mnist"
MODEL_SAVE_DIR = "./checkpoints"
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Hyperparameters
NUM_CLUSTERS = 16         # number of k-means centroids (vocab size)
CENTROIDS_PATH = f"./centroids_{NUM_CLUSTERS}.npy"
IMAGE_SIZE = 28            # mnist images are 28x28
SEQ_LEN = IMAGE_SIZE * IMAGE_SIZE  # sequence length (one token per pixel)
EMBED_DIM = 16            # embedding dimension
NUM_HEADS = 2              # number of attention heads
NUM_LAYERS = 6             # number of Transformer blocks
DROPOUT = 0.1              # dropout rate
BATCH_SIZE = 64
LR = 3e-4
DEVICE = "mps" if torch.mps.is_available() else "cpu"


###############################################################################
#                         Centroid Computation                                 #
###############################################################################

def compute_and_save_centroids(num_clusters: int = NUM_CLUSTERS,
                               batch_size: int = 10000,
                               max_samples: int = 500000):
    """
    Load MNIST train set, collect RGB pixels, run MiniBatchKMeans to get centroids,
    and save to disk.

    Args:
        num_clusters (int): number of clusters (vocabulary size).
        batch_size (int): batch size for MiniBatchKMeans.
        max_samples (int): maximum number of pixels to sample for clustering (for speed).
    """
    print(f"Computing {num_clusters} centroids from MNIST train pixels...")
    # transform to convert images to tensors and resize to 28x28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ])

    # 1. Load MNIST train data (only pixel arrays, no labels)
    mnist_train = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=True,
        transform=transform
    )
    loader = DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=0)

    # 2. Collect up to max_samples pixel RGB values
    pix_list = []
    total_pixels = 0
    for imgs, _ in tqdm(loader, desc="Collecting pixels"):
        # imgs: (B, 1, 28, 28), values in [0,1]
        imgs_np = (imgs.squeeze(1).cpu().numpy() * 255).astype(np.uint8)  # shape (B,28,28)
        imgs_np = imgs_np.reshape(-1, 1)  # flatten to (B*28*28,1)
        pix_list.append(imgs_np)
        total_pixels += imgs_np.shape[0]

        if total_pixels >= max_samples:
            break

    pix_array = np.concatenate(pix_list, axis=0)
    pix_array = pix_array[:max_samples]
    print(f"Collected {pix_array.shape[0]} pixels for k-means.")

    # 3. Run MiniBatchKMeans on pixel RGB vectors
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, verbose=1)
    kmeans.fit(pix_array)
    centroids = kmeans.cluster_centers_.astype(np.uint8)  # shape (num_clusters, 3)

    # 4. Save centroids to file
    np.save(CENTROIDS_PATH, centroids)
    print(f"Saved centroids to {CENTROIDS_PATH}")


###############################################################################
#                         Dataset and Tokenization                             #
###############################################################################

class MNISTQuantized(Dataset):
    """
    A Dataset that returns quantized token sequences for MNIST images.
    Each pixel (RGB) is mapped to the nearest centroid index (token).

    The quantization step is cached to disk for faster subsequent loads.
    """

    def __init__(self, train: bool, centroids_path: str = CENTROIDS_PATH):
        """
        Args:
            train (bool): if True, load/train split; else test split.
            centroids_path (str): path to the centroids .npy file.
        """
        super().__init__()
        self.train = train
        self.centroids = np.load(centroids_path)  # shape: (NUM_CLUSTERS, 1)
        # Load raw CMNIST dataset without transforms (we'll quantize manually)
        self.raw_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR, train=self.train, download=True, transform=transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        )

        # Path to cached tokenized file
        suffix = "train" if self.train else "test"
        self.cache_path = os.path.join(TOKENIZED_DATA_DIR, f"MNIST_{suffix}_tokens.npy")

        # If cache exists, load directly; else quantize all images and save
        if os.path.exists(self.cache_path):
            print(f"Loading tokenized data from {self.cache_path}...")
            self.tokenized = np.load(self.cache_path, mmap_mode="r")
        else:
            print(f"Tokenizing MNIST {'train' if self.train else 'test'} images...")
            self.tokenized = self._quantize_and_cache()

    def _quantize_and_cache(self):
        """
        Convert each image to a 1D array of token indices by:
        - Converting to numpy uint8 grayscale values
        - Flattening to (28*28, 1)
        - Assigning each pixel to the nearest centroid index
        """
        num_images = len(self.raw_dataset)
        token_array = np.empty((num_images, SEQ_LEN), dtype=np.int64)

        # Precompute squared centroid norms for fast nearest-neighbor
        centroids = self.centroids.astype(np.int32)  # (NUM_CLUSTERS, 1)

        for idx in tqdm(range(num_images), desc="Quantizing images"):
            img, _ = self.raw_dataset[idx]
            # img is PIL Image
            img_np = np.array(img, dtype=np.uint8)  # (28,28)
            img_np = img_np.reshape(-1, 1) # flatten to (28*28, 1)

            # Compute L2 distance to each centroid: (28*28, NUM_CLUSTERS)
            # Efficient approach: expand dims
            # dist_sq = ||pix - centroids||^2 = pix^2 + centroids^2 - 2*pix·centroids
            pix_int = img_np.astype(np.int32)
            pix_norm = np.sum(pix_int ** 2, axis=1, keepdims=True)  # (28*28,1)
            cent_norm = np.sum(centroids ** 2, axis=1)  # (NUM_CLUSTERS,)
            dot = pix_int @ centroids.T  # (28*28, NUM_CLUSTERS)
            dists = pix_norm + cent_norm - 2 * dot  # (1024, NUM_CLUSTERS)
            token_ids = np.argmin(dists, axis=1)  # (1024,)
            token_array[idx] = token_ids

        # Save to .npy for caching
        np.save(self.cache_path, token_array)
        print(f"Saved tokenized data to {self.cache_path}")
        return token_array

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        """
        Returns:
            tokens (torch.LongTensor): shape (SEQ_LEN,), dtype long, values in [0, NUM_CLUSTERS)
        """
        tokens = self.tokenized[idx]
        return torch.from_numpy(tokens).long()


###############################################################################
#                             Model Definition                                 #
###############################################################################

class TransformerGPT(nn.Module):
    """
    A simple decoder-only Transformer (GPT-like) for sequence modeling.
    - Token embedding (vocab_size -> embed_dim)
    - Positional embedding (max_seq_len -> embed_dim)
    - Stacked Transformer blocks (decoder-only, causal mask)
    - Linear head to vocab_size
    """

    def __init__(self,
                 vocab_size: int = NUM_CLUSTERS,
                 embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS,
                 max_seq_len: int = SEQ_LEN):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True  # PyTorch >=1.12 supports batch_first
        )
        self.transformer = nn.TransformerDecoder(
            transformer_layer,
            num_layers=num_layers
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # Causal mask (do not attend to future tokens)
        # We will construct this mask on-the-fly in forward()

    def _generate_causal_mask(self, seq_len: int, device: str):
        """
        Generate a causal mask of shape (seq_len, seq_len) with float(-inf) in masked positions.
        """
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device) * float("-inf"), diagonal=1)
        return mask  # (seq_len, seq_len)

    def forward(self, input_ids: torch.LongTensor):
        """
        Args:
            input_ids: (B, S) token indices

        Returns:
            logits: (B, S, vocab_size)
        """
        B, S = input_ids.size()
        device = input_ids.device

        # 1. Embed tokens and positions
        positions = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)  # (B, S)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)  # (B, S, embed_dim)

        # 2. Create causal mask
        causal_mask = self._generate_causal_mask(S, device=device)  # (S, S)

        # 3. Transformer decoding (treat input as both tgt and memory for simplicity)
        # Using transformer decoder with input as target, and no encoder memory
        # To mimic decoder-only, we pass tgt=x, memory=None, and use mask
        # Note: PyTorch's TransformerDecoder requires a memory; we'll pass x as memory
        out = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask
        )  # (B, S, embed_dim)

        # 4. Final layer norm and head
        out = self.ln_f(out)  # (B, S, embed_dim)
        logits = self.head(out)  # (B, S, vocab_size)
        return logits


###############################################################################
#                             Training Loop                                    #
###############################################################################

def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int = 50,
          save_interval: int = 10):
    """
    Training loop for generative pretraining.
    Autoregressively model next-token prediction with CrossEntropyLoss.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            # batch: (B, SEQ_LEN)
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]    # (B, SEQ_LEN-1)
            targets = batch[:, 1:]    # (B, SEQ_LEN-1)

            optimizer.zero_grad()
            logits = model(inputs)    # (B, SEQ_LEN-1, vocab_size)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )  # flatten both: (B*(SEQ_LEN-1), vocab_size) vs (B*(SEQ_LEN-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Scheduler step (if any)
        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # Save checkpoint periodically
        if epoch % save_interval == 0 or epoch == epochs:
            save_path = os.path.join(MODEL_SAVE_DIR, f"image_gpt_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")


###############################################################################
#                                Main                                          #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Image GPT for MNIST")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["compute_centroids", "train"],
        help="Mode: compute_centroids or train"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (train mode only)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for training (train mode only)"
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help="Learning rate (train mode only)"
    )
    args = parser.parse_args()

    if args.mode == "compute_centroids":
        compute_and_save_centroids()
        return

    elif args.mode == "train":
        # 1. Ensure centroids exist
        if not os.path.exists(CENTROIDS_PATH):
            raise FileNotFoundError(
                f"Centroids file not found at {CENTROIDS_PATH}. "
                "Run with --mode compute_centroids first."
            )

        # 2. Prepare Dataset and DataLoader
        train_dataset = MNISTQuantized(train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # 3. Build model and optimizer
        model = TransformerGPT().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        # Optionally a scheduler; here we use a simple cosine schedule
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # 4. Train
        train(model, train_loader, optimizer, scheduler, epochs=args.epochs)

    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
