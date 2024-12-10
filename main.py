#MAIN 

import argparse
import matplotlib.pyplot as plt

import scvi
from anndata import AnnData
from pytorch_lightning.loggers import WandbLogger
from scvi_perso import SimpleVAEModel
from scviGMvae import GMVAEModel
from visualization import visu_latent_rep
from Task_eval import clustering_eval, evaluate_imputation
from utils import corrupt_dataset


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        "--data",
        type=AnnData,
        default=None,
        metavar="D",
        help="DataSet (e.g cortex)",
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=[10, 10],
        metavar="LD",
        help="Dimension of latent dimension for the model",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["simple_vae", "gm_vae"],
        metavar="MOD",
        help="Name of the model",
    )
    parser.add_argument(
        "--eval",
        type=str_to_bool,  
        default=True,
        metavar="CO",
        help="Evaluation Mode",
    )
    parser.add_argument(
        "--training",
        type=str_to_bool,  
        default=True,
        metavar="TR",
        help="Train the model or use an already trained model",
    )
    parser.add_argument(
        "--model_saves",
        type=str,
        nargs="+",
        default=[None, None], 
        metavar="MS",
        help="Emplacement of the save model or the emplacement where you want to save the model(if Training=True)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        nargs="+",
        default=[20, 20],
        metavar="ME",
        help="Maximum epochs to train the model",
    )
    parser.add_argument(
        "--cluster_type",
        type=str,
        default="cell_type",
        metavar="CLU",
        help="Type of clusters: 'cell_type' or 'precise_labels'",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        metavar="ACC",
        help="Accelerator for training",
    )
    parser.add_argument(
        "--use_wandb",
        type=str_to_bool,
        default=True,
        metavar="WB",
        help="Use Weights & Biases (wandb) for logging",
    )
    args = parser.parse_args()
    return args

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
    

def main():
    """Main function for training and evaluation."""
    args = opts()

    if args.data == None:
        print(f"Importing cortex dataset...")
        adata = scvi.data.cortex()
        print(f"cortex dataset succesfully imported.")
    else : 
        adata = args.data

    for model_name, latent_dim, model_save, max_epochs in zip(
        args.model_names, args.latent_dims, args.model_saves, args.max_epochs or [None] * len(args.model_names)
    ):
        print(f"{model_name} with {latent_dim} latent dim")

        wandb_logger = None
        if args.use_wandb:
            wandb_logger = WandbLogger(
                project='PGM-single-cell',
                name=f"{model_name}_latent{latent_dim}_epochs{max_epochs}",
                log_model=True
            )

        if model_name == "simple_vae":
            SimpleVAEModel.setup_anndata(adata)
            model = SimpleVAEModel(adata, n_latent=latent_dim)
        elif model_name == "gm_vae":
            GMVAEModel.setup_anndata(adata)
            n_clusters = len(adata.obs[args.cluster_type].unique())
            model = GMVAEModel(adata, n_clusters=n_clusters ,n_latent=latent_dim)
        else:
            raise ValueError(f"Unknown model : {model_name}, try with simple_vae or gm_vae.")


        if args.training:
            print(f"Training {model_name}...")
            model.train(
                max_epochs=max_epochs, 
                logger=wandb_logger, 
                accelerator=args.accelerator,
                train_size=0.85,
                validation_size=0,
                )
            print(f"Model {model_name} train with success (elbo={model.get_elbo().item()}).")
            if model_save != None:
                model.save(model_save,overwrite=True)
                print(f"Model saved at : {model_save}")
        else:
            if model_save:
                if model_name == "simple_vae":
                    model = SimpleVAEModel.load(model_save, adata=adata)
                    print(f"Model {model_name} succesfully charged from : {model_save}")
                elif model_name == "gm_vae":
                    model = GMVAEModel.load(model_save, adata=adata)
                    print(f"Model {model_name} succesfully charged from : {model_save}")
            else:
                raise ValueError("No model given")

        # Evaluation
        if args.eval:
            print(f"Comparison mode for {model_name}")
            print("Visualization Clustering")
            visu_latent_rep(adata,model,save=True,rep_save=model_save, col=args.cluster_type)
            plt.show()
            print("-" * 50)
            print("Clustering Eval")
            clustering_eval(adata,model, style=args.cluster_type)
            print("-" * 50)
            print("Imputation Eval")
            test_idx = model.trainer.datamodule.test_idx 
            test_adata = adata[test_idx, :].copy()
            corrupt, mask = corrupt_dataset(test_adata.X)
            L1_error = evaluate_imputation(test_adata.X,corrupt,mask,model, model_name)
            print(f"The final L1 error is: {L1_error}")

        print("-" * 50)
        if args.use_wandb:
            wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
