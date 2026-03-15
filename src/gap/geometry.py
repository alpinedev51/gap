import torch


def distance_to_manifold(point, manifold_data):
    """
    Calculates the Euclidean distance from a single point (1x3)
    to the closest point in the manifold dataset (Nx3).
    """
    dists = torch.norm(manifold_data - point, dim=1)
    min_dist, idx = torch.min(dists, dim=0)
    return min_dist.item(), manifold_data[idx]
