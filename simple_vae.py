# Implementation of scVI [2018, Romain Lopez, Jeffrey Regier, Michael B. Cole, Michael I. Jordan and Nir Yosef]

#Import 
import torch 
import scvi
from torch import nn
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField, NumericalObsField
from torch.distributions import Normal, NegativeBinomial , Poisson
from torch.distributions import kl_divergence as kl
import torch.nn.functional as F


###########################
###########################
###### ARCHITECHTURE ######
###########################
###########################

    
# DECODER (NN5 & NN6)
class SimpleDecoder(nn.Module):
    def __init__(self, n_latent: int, n_output: int, likelihood: str, n_hidden: int = 128):
        super().__init__()
        self.likelihood = likelihood.lower()
        self.n_latent = n_latent
        self.n_output = n_output
        
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(p=0.1)
        
        if self.likelihood in ['nb', 'zinb']:
            self.output_mean = nn.Linear(n_hidden, n_output)
            self.output_disp = nn.Linear(n_hidden, n_output)
            if self.likelihood == 'zinb':
                self.output_zero_prob = nn.Linear(n_hidden, n_output)
        elif self.likelihood in ['p', 'zip']:
            self.output_lambda = nn.Linear(n_hidden, n_output)
            if self.likelihood == 'zip':
                self.output_zero_prob = nn.Linear(n_hidden, n_output)
        else:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")

    def forward(self, z: torch.Tensor):
        h = self.dropout(F.relu(self.bn1(self.fc1(z))))
        h = self.dropout(F.relu(self.bn2(self.fc2(h))))
        
        if self.likelihood == 'nb':
            mean = F.softplus(self.output_mean(h))
            disp = F.softplus(self.output_disp(h))
            return {'mean': mean, 'disp': disp}
        
        elif self.likelihood == 'zinb':
            mean = F.softplus(self.output_mean(h))
            disp = F.softplus(self.output_disp(h))
            zero_prob = torch.sigmoid(self.output_zero_prob(h))
            return {'mean': mean, 'disp': disp, 'zero_prob': zero_prob}
        
        elif self.likelihood == 'p':
            lambda_ = F.softplus(self.output_lambda(h))
            return {'lambda': lambda_}
        
        elif self.likelihood == 'zip':
            lambda_ = F.softplus(self.output_lambda(h))
            zero_prob = torch.sigmoid(self.output_zero_prob(h))
            return {'lambda': lambda_, 'zero_prob': zero_prob}
        
        else:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")

#ENCODER (NN3 & NN4)
class SimpleEncoder(nn.Module):
    def __init__(self, n_input: int, n_latent: int, n_hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.mean_layer = nn.Linear(n_hidden, n_latent)
        self.var_layer = nn.Linear(n_hidden, n_latent)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        h = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        h = self.dropout(torch.relu(self.bn2(self.fc2(h))))
        mean = self.mean_layer(h)
        log_var = self.var_layer(h)
        return mean, log_var




###########################
###########################
########## MODEL ##########
###########################
###########################


class SimpleVAEModule(BaseModuleClass):
    """Simple Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_latent
        Dimensionality of the latent space.
    """

    def __init__(
        self,
        n_input: int,
        likelihood: str,
        n_latent: int = 10,
    ):
        super().__init__()
        self.likelihood = likelihood.lower()
        self.encoder = SimpleEncoder(n_input, n_latent)
        self.decoder = SimpleDecoder(n_latent, n_input, likelihood=self.likelihood)


    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Parse the dictionary to get appropriate args"""
        # let us fetch the raw counts, and add them to the dictionary
        return {"x": tensors[REGISTRY_KEYS.X_KEY]}

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = torch.log1p(x)
        qz_m, qz_v_log = self.encoder(x_)
        qz_v = qz_v_log.exp()
        z = Normal(qz_m, torch.sqrt(qz_v)).rsample()

        return {"qzm": qz_m, "qzv": qz_v, "z": z}

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "z": inference_outputs["z"],
            # "library": torch.sum(tensors[REGISTRY_KEYS.X_KEY], dim=1, keepdim=True),
        }

    @auto_move_data
    def generative(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the generative model."""
        generative_outputs = self.decoder(z)
        return generative_outputs

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> LossOutput:
        x = tensors[REGISTRY_KEYS.X_KEY]
        
        if self.likelihood == 'nb':
            nb_mean = generative_outputs["mean"]
            nb_disp = generative_outputs["disp"]
            log_likelihood = NegativeBinomial(total_count=nb_disp, 
                                              logits=torch.log((nb_disp / nb_mean) + 1e-4)).log_prob(x).sum(dim=-1)
        
        elif self.likelihood == 'zinb':
            nb_mean = generative_outputs["mean"]
            nb_disp = generative_outputs["disp"]
            zero_prob = generative_outputs["zero_prob"]
            nb_dist = NegativeBinomial(total_count=nb_disp, logits=torch.log((nb_disp / nb_mean) + 1e-4))
            log_nb_prob = nb_dist.log_prob(x)
            log_nb_prob_zero = nb_dist.log_prob(torch.zeros_like(x))
            log_likelihood = torch.where(
                x == 0,
                torch.log(torch.sigmoid(zero_prob) + torch.exp(log_nb_prob_zero) * (1 - torch.sigmoid(zero_prob)) + 1e-10),
                torch.log(1 - torch.sigmoid(zero_prob)) + log_nb_prob
            ).sum(dim=-1)
        
        elif self.likelihood == 'p':
            lambda_ = generative_outputs["lambda"]
            poisson_dist = Poisson(rate=lambda_)
            log_likelihood = poisson_dist.log_prob(x).sum(dim=-1)
        
        elif self.likelihood == 'zip':
            lambda_ = generative_outputs["lambda"]
            zero_prob = generative_outputs["zero_prob"]
            poisson_dist = Poisson(rate=lambda_)
            log_poisson_prob = poisson_dist.log_prob(x)
            log_poisson_prob_zero = poisson_dist.log_prob(torch.zeros_like(x))
            log_likelihood = torch.where(
                x == 0,
                torch.log(torch.sigmoid(zero_prob) + torch.exp(log_poisson_prob_zero) * (1 - torch.sigmoid(zero_prob)) + 1e-10),
                torch.log(1 - torch.sigmoid(zero_prob)) + log_poisson_prob
            ).sum(dim=-1)
        
        else:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")

        qz_m = inference_outputs["qzm"]
        qz_v = inference_outputs["qzv"]

        prior_dist = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=-1)

        elbo = log_likelihood - kl_divergence
        loss = torch.mean(-elbo)
        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_likelihood,
            kl_local=kl_divergence,
        )


class SimpleVAEModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """Single-cell Variational Inference with multiple likelihoods."""

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        likelihood: str = 'nb',
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = SimpleVAEModule(
            n_input=self.summary_stats["n_vars"],
            n_latent=n_latent,
            likelihood=likelihood,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"SCVI Model with the following params: \n"
            f"n_latent: {n_latent}, likelihood: {likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)




