import torch

# function to compute squared euclidian distance
def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdim=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdim=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d

# function to quantize a single image as a string of tokens based on centroids
def quantize(img, centroids):
    img = img.permute(1,2, 0).contiguous()
    img = img.view(-1, 1) # flatten to pixels

    # calc distance to centroids
    d = squared_euclidean_distance(img, centroids)
    tokens = torch.argmin(d, 1) # choose closest centroid index for each pixel
    return tokens

# unquantize tokens to pixel values using centroids
def unquantize(tokens, centroids):
    return centroids[tokens]