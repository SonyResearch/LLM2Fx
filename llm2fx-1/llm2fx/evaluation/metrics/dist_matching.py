import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian kernel between two vectors"""
    dist = euclidean_distances(x, y)
    return np.exp(-dist / (2 * sigma * sigma))

def compute_similarity_stats(x, y):
    sim_distribution = cosine_similarity(x, y)
    np.fill_diagonal(sim_distribution, 0)
    n = sim_distribution.shape[0]
    avg_sim = np.mean(sim_distribution)
    std_sim = np.std(sim_distribution)
    return avg_sim, std_sim

def compute_mmd(x, y, sigma=1.0, coeff=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples
    Args:
        x: First set of samples (n_samples_1 x n_features)
        y: Second set of samples (n_samples_2 x n_features)
        sigma: Bandwidth parameter for Gaussian kernel
    Returns:
        MMD value between the two sample sets
    """
    x = np.array(x)
    y = np.array(y)
    # Compute kernel matrices
    k_xx = gaussian_kernel(x, x, sigma)
    k_yy = gaussian_kernel(y, y, sigma)
    k_xy = gaussian_kernel(x, y, sigma)
    # Compute MMD
    n_x = x.shape[0]
    n_y = y.shape[0]
    mmd = (np.sum(k_xx) / (n_x * n_x) +
           np.sum(k_yy) / (n_y * n_y) -
           2 * np.sum(k_xy) / (n_x * n_y))
    mmd = np.sqrt(max(mmd, 0)) * coeff
    return mmd
