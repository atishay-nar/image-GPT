import argparse
import yaml
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

# reducing size of images, classes, and number of samples per class
class ReducedMNISTDataset(Dataset):
    def __init__(self, cfg, train=True):
        super().__init__()
        #
        classes = cfg.CLASSES
        amts = {}
        for c in classes:
            amts[c] = cfg.SAMPLES_PER_CLASS
            if not train:
                amts[c] /= 6  # sixth of the amount for test set

        transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),  # <-- resize
        transforms.ToTensor(),                    # <-- to tensor
        ])

        # load raw data
        raw_dataset = torchvision.datasets.MNIST(root=cfg.DATA_DIR, train=True, download=True, transform=transform)
        self.reduced = self.reduce_data(raw_dataset, amts)
    
    def reduce_data(self, raw_dataset, amts):
        reduced = []
        # filter
        for itm in raw_dataset:
            img, label = itm
            if label in amts and amts[label] > 0:
                reduced.append(img)
                amts[label] -= 1
    
        return reduced

    # needed for Dataset class
    def __len__(self):
        return len(self.reduced)
    
    # needed for Dataset class
    def __getitem__(self, idx):
        img = self.reduced[idx]
        return img

if __name__ == "__main__":
    cfg = argparse.Namespace(**yaml.safe_load(open("configs.yml", "r")))
    test = ReducedMNISTDataset(cfg)
    print(test[0].shape)