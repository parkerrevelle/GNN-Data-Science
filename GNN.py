import torch
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.measure import regionprops
import numpy as np
from sympy.physics.control.control_plots import plt
from torchvision.transforms import ToTensor

def image_to_graph(image_path):
    # Read and normalize the image
    image = plt.imread(image_path)
    image = image / image.max()
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    # Compute region properties to extract features
    regions = regionprops(segments, intensity_image=rgb2gray(image))
    features = [np.array([prop.area, prop.mean_intensity]) for prop in regions]
    node_features = torch.tensor(features, dtype=torch.float)

    # Build edges (connect each node to its neighbor)
    adjacency = np.zeros((len(regions), len(regions)), dtype=int)
    for props in regions:
        for coord in props.coords:
            i, j = coord
            if i > 0 and segments[i - 1, j] != props.label:
                adjacency[props.label, segments[i - 1, j]] = 1
            if i < segments.shape[0] - 1 and segments[i + 1, j] != props.label:
                adjacency[props.label, segments[i + 1, j]] = 1
            if j > 0 and segments[i, j - 1] != props.label:
                adjacency[props.label, segments[i, j - 1]] = 1
            if j < segments.shape[1] - 1 and segments[i, j + 1] != props.label:
                adjacency[props.label, segments[i, j + 1]] = 1

    # Convert adjacency matrix to edge index format
    edge_index = adjacency.nonzero()
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return node_features, edge_index

