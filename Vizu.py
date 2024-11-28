#Vizualization for scVI models


#Import 
import scanpy as sc
import os
import torch
import numpy as np

from anndata import AnnData
#from scvi_perso import SimpleVAEModel, SimpleVAEModule
from scvi.model.base import BaseModelClass
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from scviGMvae import GMVAEModel
 

#Utils 

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


#VIZU


def vizu_latent_rep(data : AnnData, model : BaseModelClass, save : bool = False, precise : bool = False, rep_save : str = None):
    """
    Visualize the effectiveness of the latent projection of the model
    
    """

    if not(model.is_trained) :
        raise ValueError("Your model is not trained.")
    
    if precise : 
        col = "precise_labels"
    else : 
        col = "cell_type"

    adata = data.copy()
    if type(model) == GMVAEModel : 
        inference = model.module.inference(torch.tensor(data.X))
        qzm = inference["qzm"]  
        qzv = inference["qzv"] 
        probs_y = inference["probs_y"]  

        latent = sample_from_gmm(qzm, qzv, probs_y).detach().numpy()
    else : 
        latent = model.get_latent_representation(adata)
    latent_data = AnnData(X = latent, obs = adata.obs)

    #Compute Umap for latent representation
    sc.pp.neighbors(latent_data, n_pcs=30, n_neighbors=20)
    sc.tl.umap(latent_data, min_dist=0.3)

    #Compute Umap for raw data (with PCA)
    sc.tl.pca(adata)#, n_comps=latent.shape[1])
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=20)
    sc.tl.umap(adata, min_dist=0.3)

    #Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) 
    sc.pl.umap(
        latent_data,
        color=[col],
        ax=axes[0],  
        show=False  
    )
    axes[0].set_title("UMAP for the latent representation", fontsize=16)

    
    sc.pl.umap(
        adata,
        color=[col],  
        ax=axes[1],  # Passer le subplot
        show=False
    )
    axes[1].set_title("UMAP for Raw Data", fontsize=16)

    plt.tight_layout()
    if save : 
        save_path = f"{rep_save}/img/Model_Latent.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Image save in {rep_save}/img")
    plt.show()







