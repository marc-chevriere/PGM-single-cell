# UTILS


#IMPORT
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from torch.distributions import Normal, Categorical

#CORRUPT

def corrupt_dataset(data : np.array, method="uniform_zero", corruption_rate=0.1):
    """
    Apply a corruption to the dataset as in [2018Lopez].

    """
    corrupted_data = data.copy()
    mask = np.zeros_like(data, dtype=bool)
    
    # Indice Selection
    nonzero_indices = np.argwhere(data > 0)
    num_corruptions = int(len(nonzero_indices) * corruption_rate)
    selected_indices = nonzero_indices[np.random.choice(len(nonzero_indices), num_corruptions, replace=False)]
    
    if method == "uniform_zero":
        for i, j in selected_indices:
            corrupted_data[i, j] *= np.random.binomial(1, 0.9)
    elif method == "binomial":
        for i, j in selected_indices:
            corrupted_data[i, j] = np.random.binomial(data[i, j], 0.2)
    else:
        raise ValueError("Invalid corruption method. Choose 'uniform_zero' or 'binomial'.")
    
    # Mettre à jour le masque
    mask[tuple(zip(*selected_indices))] = True
    
    return corrupted_data, mask


#Cluster acc

def cluster_accuracy(true_labels, predicted_labels):
    true_labels = list(map(int, true_labels))
    contingency_matrix = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / contingency_matrix.sum()
    return accuracy


#Sample from gmm

def sample_from_gmm(qzm, qzv, probs_y):
    """
    Sample points from a Gaussian Mixture Model (GMM).
    """

    n_samples, n_clusters, n_latent = qzm.shape

    categorical_dist = Categorical(probs_y)  
    sampled_categories = categorical_dist.sample() 

    selected_means = qzm[torch.arange(n_samples), sampled_categories]  # (n_samples, n_latent)
    selected_vars = qzv[torch.arange(n_samples), sampled_categories]  # (n_samples, n_latent)

    normal_dist = Normal(selected_means, torch.sqrt(selected_vars))
    samples = normal_dist.rsample()  # (n_samples, n_latent)

    return samples

