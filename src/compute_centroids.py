# imports
import argparse
import yaml
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from reduced_mnist import ReducedMNISTDataset

        
# set random seed
np.random.seed(37)

# centroid calculation function
def compute_centroids(cfg):

    # load MNIST train data (only pixel arrays, no labels)
    reduced_mnist = ReducedMNISTDataset(cfg) 

    # transform into pixel array
    train_x = np.stack([x.numpy()for (x, _) in reduced_mnist])
    train_x = train_x.transpose(0, 2, 3, 1)
    
    pixels = train_x.reshape(-1, train_x.shape[-1])

    # perform k-means clustering
    kmeans = MiniBatchKMeans(n_clusters=cfg.NUM_CLUSTERS, batch_size=8192, verbose=1).fit(pixels)
    centroids = kmeans.cluster_centers_  # shape (num_clusters, 1)

    # Save centroids to file 
    # SOL cannot use SKLearn efficiently so it is more effective to precompute centroids on local machine
    centroid_path = os.path.join(cfg.CENTROID_DIR, f"centroids_{cfg.NUM_CLUSTERS}.npy")
    os.makedirs(cfg.CENTROID_DIR, exist_ok=True)
    np.save(centroid_path, centroids)
    print(f"Saved centroids to {centroid_path}")

# main
if __name__ == "__main__":
    dict = yaml.safe_load(open("configs.yml", "r"))
    cfg = argparse.Namespace(**dict)
    centroid = compute_centroids(cfg)