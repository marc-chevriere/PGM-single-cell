#Vizualization for scVI models


#Import 
import scanpy as sc
import os
import wandb

from anndata import AnnData
from scvi.model.base import BaseModelClass
import matplotlib.pyplot as plt
from scviGMvae import GMVAEModel 


#VIZU


def visu_latent_rep(data : AnnData, model : BaseModelClass, col : str, save : bool = False, rep_save : str = None):
    """
    Visualize the effectiveness of the latent projection of the model
    
    """

    if not(model.is_trained) :
        raise ValueError("Your model is not trained.")

    adata = data.copy()
    latent_representation = model.get_latent_representation(data)
    if type(model) == GMVAEModel : 
        latent_cat = latent_representation["latent_cat"]
        latent_representation = latent_representation["latent_rep"]
    latent_data = AnnData(X = latent_representation, obs = adata.obs)

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
        if wandb.run is not None:  # Vérifie si wandb est actif
            wandb.log({"Model Latent Representation": wandb.Image(save_path)})
    plt.show()







