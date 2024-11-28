#Vizualization for scVI models


#Import 
import scanpy as sc
from anndata import AnnData
#from scvi_perso import SimpleVAEModel, SimpleVAEModule
from scvi.model.base import BaseModelClass
import matplotlib.pyplot as plt


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
        plt.savefig(f"{rep_save}/img/Model_Latent.png")
    plt.show()







