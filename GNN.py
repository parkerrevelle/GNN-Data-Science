import torch
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.measure import regionprops
import numpy as np
from torchvision.transforms import ToTensor

def image_to_graph(image_path):
    # Read and normalize the image
    image = plt.imread(image_path)
    image = image / image.max()
    segments = slic(image, n_segments=100, compactness=10, sigma=1)