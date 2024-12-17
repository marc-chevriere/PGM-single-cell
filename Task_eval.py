#Evalutation of scVI models

#IMPORT 
from anndata import AnnData
from scvi.model.base import BaseModelClass
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
import numpy as np
import torch

from utils import cluster_accuracy
from gm_vae import GMVAEModel


###########################
###########################
####### CLUSTERING ########
###########################
###########################


def clustering_eval(data : AnnData, model : BaseModelClass, style : str):     
    """
    Evaluate Clustering task for scVI models.

    """

    true_labels = (data.obs[style]).to_numpy()

    latent_representation = model.get_latent_representation(data)
    if type(model) == GMVAEModel : 
        latent_cat = latent_representation["latent_cat"]
        latent_representation = latent_representation["latent_rep"]
        predicted_labels = np.argmax(latent_cat, axis=-1)
    else:
        #Kmeans on the latent representation
        kmeans = KMeans(n_clusters=len(set(true_labels)), n_init=200, random_state=42)
        predicted_labels = kmeans.fit_predict(latent_representation)

    #Various ways of evaluating the clustering
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(latent_representation, predicted_labels)
    homogeneity = homogeneity_score(true_labels, predicted_labels)
    completeness = completeness_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    accuracy = cluster_accuracy(true_labels, predicted_labels)

    print("VAE :")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Homogeneity Score: {homogeneity:.4f}")
    print(f"  Completeness Score: {completeness:.4f}")
    print(f"  Cluster Accuracy: {completeness:.4f}")

    return ari, nmi, silhouette, homogeneity, completeness, v_measure, accuracy




###########################
###########################
####### IMPUTATION ########
###########################
###########################


def evaluate_imputation(data : np.array, corrupted_data : np.array, 
                        mask : np.array, model : BaseModelClass, model_name: str,
                        likelihood: str):
    """
    Evaluate Imputation task for scVI models.

    """
    corrupted_data_tensor = torch.tensor(corrupted_data, dtype=torch.float32)
    
    with torch.no_grad():
        inference_outputs = model.module.inference(corrupted_data_tensor)
        z = inference_outputs["z"]
        generative_outputs = model.module.generative(z)
        if likelihood == 'nb':
            imputed_values = generative_outputs["mean"]
        
        elif likelihood == 'zinb':
            nb_mean = generative_outputs["mean"]
            zero_prob = generative_outputs["zero_prob"]
            pi = torch.sigmoid(zero_prob)  
            imputed_values = (1 - pi) * nb_mean
        
        elif likelihood == 'p':
            imputed_values = generative_outputs["lambda"]
        
        elif likelihood == 'zip':
            lambda_ = generative_outputs["lambda"]
            zero_prob = generative_outputs["zero_prob"]
            pi = torch.sigmoid(zero_prob) 
            imputed_values = (1 - pi) * lambda_
        
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")

        if model_name=="gm_vae":
            probs_y = inference_outputs["probs_y"]
            imputed_values = (imputed_values * probs_y.unsqueeze(-1)).sum(dim=1)
        imputed_values = imputed_values.cpu().numpy()

    # L1 distance for corrupted data
    l1_distances_corrupted = np.abs(corrupted_data_tensor[mask] - imputed_values[mask])
    l1_distances = np.abs(data[mask] - imputed_values[mask])
    median_l1 = np.median(l1_distances)
    median_l1_corrupted = torch.median(l1_distances_corrupted).item()
    mean_l1 = np.mean(l1_distances)
    mean_l1_corrupted = torch.mean(l1_distances_corrupted).item()
    
    return median_l1, median_l1_corrupted, mean_l1, mean_l1_corrupted

