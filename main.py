#MAIN 

import argparse
import os
import scvi

from scvi_perso import SimpleVAEModel
from scviGMvae import GMVAEModel
from Vizu import vizu_latent_rep
from Task_eval import clustering_eval, corrupt_dataset, evaluate_imputation


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--data",
        type=str,
        default="Cortex",
        metavar="D",
        help="Type of data",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[10],
        metavar="LD",
        help="Dimension of latent dimension for the model",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default="",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--comparison",
        type=bool,
        default=True,
        metavar="CO",
        help="Comparison Mode",
    )
    parser.add_argument(
        "--training",
        type=bool,
        default=False,
        metavar="TR",
        help="Train the model or you have already a train model",
    )
    parser.add_argument(
        "--model_save",
        type=str,
        nargs="+",
        default=None,
        metavar="MS",
        help="Emplacement and Names of the save model or the emplacement where you want to save the model",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        nargs="+",
        default=None,
        metavar="ME",
        help="Maximum epochs to train the model",
    )
    args = parser.parse_args()
    return args


def main():
    """ Default main function """

    args = opts()

    if args.data == "cortex":
        adata = scvi.data.cortex()

    n_latent = args.latent_dim
    for name_models in args.model_names : 
        if args.Training : 
            if name_models == "simple_vae":
                SimpleVAEModel.setup_anndata(adata)
                simple_vae = SimpleVAEModel(adata, n_latent=n_latent)
                
def main():
    """Main function for training and evaluation."""
    args = opts()


    if args.data == "cortex":
        adata = scvi.data.cortex()


    for model_name, latent_dim, model_save, max_epochs in zip(
        args.model_names, args.latent_dims, args.model_saves, args.max_epochs or [None] * len(args.model_names)
    ):
        print(f" {model_name} with {latent_dim} latent dim")


        if model_name == "SimpleVAE":
            model = SimpleVAEModel(adata, n_latent=latent_dim)
        elif model_name == "GMVAE":
            model = GMVAEModel(adata, n_latent=latent_dim)
        else:
            raise ValueError(f"Unknown model : {model_name}")


        if args.training:
            model.train(max_epochs=max_epochs, logger=None)
            print(f"Model {model_name} train with success.")
            if model_save:
                model.save(model_save)
                print(f"Model saved at : {model_save}")
        else:
            if model_save:
                model = scvi.model.SCVI.load(model_save, adata=adata)
                print(f"Model {model_name} succesfully charged from : {model_save}")
            else:
                raise ValueError("No model given")

        # Évaluation et comparaison
        if args.comparison:
            print(f"Comparison mode for {model_name}")
            print("Visualization CLustering")
            vizu_latent_rep(adata,model)
            print("-" * 50)
            print("Clustering Eval")
            clustering_eval(adata,model)
            print("-" * 50)
            print("Imputation Eval")
            corrupt, mask = corrupt_dataset(adata)
            L1_error = evaluate_imputation(adata,corrupt,mask)
            print(f"The L1 error is: {L1_error}")

        print("-" * 50)


if __name__ == "__main__":
    main()
