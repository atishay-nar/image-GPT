# imports
import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

# set device
DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
        )
        
# set random seed
np.random.seed(0)

# class that transforms dataset, here MNIT, into a tokenized array based on the centroids
class TokenizedData(Dataset):
    
    def __init__(self, cfg):
        super().__init__()

        # Where to cache tokenized data
        os.makedirs(cfg.TOKENIZED_DATA_DIR, exist_ok=True)
        self.cache_path = os.path.join(cfg.TOKENIZED_DATA_DIR, f"MNIST_tokens_{cfg.NUM_CLUSTERS}.npy")

        # If cache exists, load directly; otherwise quantize all images and save
        if os.path.exists(self.cache_path):
            self.tokenized = np.load(self.cache_path, mmap_mode="r").copy()
        else:
            self.SEQ_LEN = cfg.IMAGE_SIZE ** 2
            # Load raw MNIST dataset 
            raw_dataset = torchvision.datasets.MNIST(root=cfg.DATA_DIR, train=True, download=True, transform=transforms.ToTensor())

            # get centroids
            centroids_path = os.path.join(cfg.CENTROID_DIR, f"centroids_{cfg.NUM_CLUSTERS}.npy")
            self.centroids = torch.tensor(np.load(centroids_path)).to(DEVICE)
            self.tokenized = self.quantize(raw_dataset, self.centroids)

    # compute distance
    def squared_euclidean_distance(self, a, b):
        b = torch.transpose(b, 0, 1)
        a2 = torch.sum(torch.square(a), dim=1, keepdim=True)
        b2 = torch.sum(torch.square(b), dim=0, keepdim=True)
        ab = torch.matmul(a, b)
        d = a2 - 2 * ab + b2
        return d
    
    # quantize images
    def quantize(self, pre_images, centroids):
        # create empty array to hold quantized images
        token_array = np.empty((len(pre_images), self.SEQ_LEN), dtype=np.int32)
        # loop through images
        i = 0
        for itm in tqdm(pre_images, desc="Quantizing Images"):
            # get image and turn into list of pixels
            img = itm[0].detach().clone().to(DEVICE) # create image tensor
            img = img.permute(1,2, 0).contiguous()
            img = img.view(-1, 1) # flatten to pixels

            # calc distance to centroids
            d = self.squared_euclidean_distance(img, centroids)
            tokens = torch.argmin(d, 1) # replace with closest centroid index for each pixel

            token_array[i] = tokens.cpu().numpy() # store in array
            i += 1

        # save tokenized data to cache
        np.save(self.cache_path, token_array)
        print(f"Saved tokenized data to {self.cache_path}")
        return token_array
    # needed for Dataset
    def __len__(self):
        return len(self.tokenized)
    
    # needed for Dataset
    def __getitem__(self, idx):
        tokens = self.tokenized[idx]
        return torch.from_numpy(tokens)

            
if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    test = TokenizedData(cfg)


    