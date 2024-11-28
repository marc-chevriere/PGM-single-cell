#Evalutation of scVI models

#IMPORT 
from anndata import AnnData
from scvi.model.base import BaseModelClass
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, confusion_matrix
from sklearn.cluster import KMeans
import numpy as np
import torch

from utils import cluster_accuracy,sample_from_gmm
from scviGMvae import GMVAEModel


###########################
###########################
####### CLUSTERING ########
###########################
###########################


def clustering_eval(data : AnnData, model : BaseModelClass, precise : bool = True):     
    """
    Evaluate Clustering task for scVI models.

    """
    if precise : 
        style = "precise_labels"
    else : 
        style = "cell_type"

    true_labels = (data.obs[style]).to_numpy()

    if type(model) == GMVAEModel : 
        inference = model.module.inference(torch.tensor(data.X))
        qzm = inference["qzm"]  
        qzv = inference["qzv"] 
        probs_y = inference["probs_y"]  

        latent_representation = sample_from_gmm(qzm, qzv, probs_y).detach().numpy()
    else : 
        latent_representation = model.get_latent_representation(data)

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

    print("Simple VAE :")
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


def evaluate_imputation(data : np.array, corrupted_data : np.array, mask : np.array, model : BaseModelClass):
    """
    Evaluate Imputation task for scVI models.

    """

    corrupted_data_tensor = torch.tensor(corrupted_data, dtype=torch.float32)
    
    with torch.no_grad():
        inference_outputs = model.module.inference(corrupted_data_tensor)
        z = inference_outputs["z"]  
        generative_outputs = model.module.generative(z)
        imputed_values = generative_outputs["nb_mean"].numpy() 

    # L1 distance for corrupted data
    l1_distances = np.abs(data[mask] - imputed_values[mask])
    median_l1 = np.median(l1_distances)
    
    return median_l1

