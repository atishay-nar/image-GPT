import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from image_gpt import ImageGPT
from utils import quantize, unquantize

def sample(cfg):
    return

if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    sample(cfg)
    