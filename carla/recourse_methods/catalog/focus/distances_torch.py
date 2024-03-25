import torch
import torch.nn.functional as F


def distance_func(name, x1, x2, eps: float = 0.0):
    if name == "l1":
        ax = 1
        return l1_dist(x1, x2, ax, eps)
    if name == "l2":
        ax = 1
        return l2_dist(x1, x2, ax, eps)
    if name == "cosine":
        ax = 1  # Note: PyTorch calculates cosine similarity, not distance, and does so over dimensions differently.
        return cosine_dist(x1, x2, ax, eps)


def l1_dist(x1, x2, ax: int, eps: float = 0.0):
    # sum over |x| + eps, i.e. L1 norm
    x = x1 - x2
    return torch.sum(torch.abs(x), dim=ax) + eps


def l2_dist(x1, x2, ax: int, eps: float = 0.0):
    # sqrt((sum over x^2) + eps)), i.e. L2 norm
    x = x1 - x2
    return torch.sqrt(torch.sum(x**2, dim=ax) + eps)


def cosine_dist(x1, x2, ax: int, eps: float = 0.0):
    # PyTorch calculates cosine similarity, and we convert it to distance
    # Note: Cosine similarity ranges from -1 to 1, closer to 1 means more similar.
    # To convert similarity to distance, you can use 1 - similarity.
    # No need to normalize inputs; F.cosine_similarity does this internally.
    cosine_similarity = F.cosine_similarity(
        x1, x2, dim=ax
    )  # Outputs range from -1 to 1
    dist = (
        1 - cosine_similarity + eps
    )  # Convert to distance, closer to 0 means more similar
    return dist
