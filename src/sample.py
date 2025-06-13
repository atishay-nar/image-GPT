import argparse
import yaml
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from image_gpt import ImageGPT
from utils import quantize, unquantize

# set device
DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
        )

# set random seed 
torch.manual_seed(0)  

# generate the next part of image given a contex
def generate(model, context, length, num_samples=1):
    output = context.unsqueeze(0).repeat(num_samples, 1)  # add batch size of num_samples so shape [seq len, batch]

    # precfg_dict
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False, desc=f"Generating"):
            logits = model(output)               # (batch, seq, vocab)
            logits = logits[:, -1, :]            # (batch, vocab)
            probs = F.softmax(logits, dim=-1)  # convert logits to probabilities
            pred = torch.multinomial(probs, num_samples=1) # sample from the distribution
            output = torch.cat((output, pred), dim=1)  # append the precfg_dicted token to the output

    return output

def sample(cfg):
    os.makedirs('./figures', exist_ok=True)
  
    # create model and load checkpoint
    ckpt = torch.load(f'./checkpoints/image_gpt_epoch_{cfg.checkpoint}.pth', map_location=DEVICE)
    model = ImageGPT(cfg).to(DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    

    # load centroids
    centroids_path = os.path.join(cfg.CENTROID_DIR, f"centroids_{cfg.NUM_CLUSTERS}.npy")
    centroids = torch.tensor(np.load(centroids_path)).to(DEVICE)

    # load dataset
    test_data = torchvision.datasets.MNIST(root=cfg.DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    loader = iter(DataLoader(test_data, batch_size=1, shuffle=True))

    # rows for each image in final figure
    rows = []
    for example in tqdm(range(cfg.num_examples), desc="Sampling Images"):
        # get random image
        img, _ = next(loader)
        img = img[0].to(DEVICE)


        # quantize image to tokens
        img = quantize(img, centroids).cpu().numpy() # get tokens and flatten into seq
        tokens = img.reshape(-1)
        img = img.reshape(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE) # for plotting
 

        # choose the context. Here, we use the first half of the image
        context = tokens[:int(len(tokens) / 2)]
        context_img = np.pad(context, (0, int(len(tokens) / 2))).reshape(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE) # for plotting
        context = torch.from_numpy(context).long().to(DEVICE)  # convert to tensor and move to device

        # generate the rest of the image from the context
        pred = generate(model, context, int(len(tokens) / 2), num_samples=cfg.num_samples).cpu().numpy()
        pred = pred.reshape(-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)  # reshape to image size

        # add example to rows
        rows.append(np.concatenate([context_img[None, ...], pred, img[None, ...]], axis=0)) # 
    
    fig = np.stack(rows, axis=0)  # stack all rows together

    nrow, ncol, h, w = fig.shape
    fig = unquantize(fig.swapaxes(1, 2).reshape(h * nrow, w * ncol), centroids).cpu().numpy()
    fig = (fig * 255).round().astype(np.uint8)
    pic = Image.fromarray(np.squeeze(fig))
    pic.save(f"./figures/sample_at_epoch_{cfg.checkpoint}.png")




        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_num", required=True, type=int)
    parser.add_argument("--n_examples", default=3, type=int)
    parser.add_argument("--n_samples", default=3, type=int)
    args, _ = parser.parse_known_args()
    cfg_dict = yaml.safe_load(open("configs.yml", "r"))
    cfg_dict["checkpoint"] = args.ckpt_num
    cfg_dict["num_examples"] = args.n_examples
    cfg_dict["num_samples"] = args.n_samples
    cfg = argparse.Namespace(**cfg_dict)
    sample(cfg)
    