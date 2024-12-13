# Implementation of GM sc-VAE [2018, Christopher H Grønbech, Maximillian F Vording...] in the scVI context

#IMPORT 
import torch 
from torch import nn, Tensor
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS
from scvi.data.fields import (CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField, NumericalObsField)
from torch.distributions import Normal, NegativeBinomial, Poisson
from torch.distributions import kl_divergence as kl
from collections.abc import Iterator, Sequence

###########################
###########################
###### ARCHITECHTURE ######
###########################
###########################


class EncoderXtoY(nn.Module):
    def __init__(self, n_input: int, n_clusters: int, n_hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden), 
            nn.ReLU(),
            nn.Dropout(p=0.1),     
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_clusters),
            nn.Softmax(dim=-1),       
        )

    def forward(self, x: torch.Tensor):
        probs_y = self.mlp(x)
        return probs_y
    


class EncoderXYtoZ(nn.Module):
    def __init__(self, n_input: int, n_clusters: int, n_latent: int, n_hidden: int = 128):
        super().__init__()
        self.proj_y = nn.Sequential(
            nn.Linear(n_clusters, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden)
        )
        self.proj_x = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden)
        )
        self.commonlayer = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.output_mean = nn.Linear(n_hidden, n_latent)
        self.output_logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        proj_x = self.proj_x(x)
        proj_y = self.proj_y(y)
        xy = torch.cat((proj_x,proj_y), dim=-1)
        h = self.commonlayer(xy)
        mean_n = self.output_mean(h)
        logvar_n = self.output_logvar(h)
        return mean_n, logvar_n
    
    
class DecoderZtoX(nn.Module):
    def __init__(self, n_output: int, n_latent: int, likelihood: str, n_hidden: int = 128):
        super().__init__()
        self.likelihood = likelihood.lower()
        self.n_latent = n_latent
        self.n_output = n_output
        self.mlp = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.BatchNorm1d(n_hidden), 
            nn.ReLU(),
            nn.Dropout(p=0.1),        
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )     
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
        h = self.mlp(z)
        
        if self.likelihood == 'nb':
            mean = torch.nn.functional.softplus(self.output_mean(h))
            disp = torch.nn.functional.softplus(self.output_disp(h))
            return {'mean': mean, 'disp': disp}
        
        elif self.likelihood == 'zinb':
            mean = torch.nn.functional.softplus(self.output_mean(h))
            disp = torch.nn.functional.softplus(self.output_disp(h))
            zero_prob = torch.sigmoid(self.output_zero_prob(h))
            return {'mean': mean, 'disp': disp, 'zero_prob': zero_prob}
        
        elif self.likelihood == 'p':
            lambda_ = torch.nn.functional.softplus(self.output_lambda(h))
            return {'lambda': lambda_}
        
        elif self.likelihood == 'zip':
            lambda_ = torch.nn.functional.softplus(self.output_lambda(h))
            zero_prob = torch.sigmoid(self.output_zero_prob(h))
            return {'lambda': lambda_, 'zero_prob': zero_prob}
    


###########################
###########################
########## MODEL ##########
###########################
###########################





class GMVAEModule(BaseModuleClass):
    """GM Variational auto-encoder model.
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
        n_clusters: int,
        likelihood: str,
        n_latent: int = 10,
    ):
        super().__init__()
        self.likelihood = likelihood.lower()
        self.encoderxtoy = EncoderXtoY(n_input=n_input, n_clusters=n_clusters)
        self.encoderxytoz = EncoderXYtoZ(n_clusters=n_clusters, n_input=n_input, n_latent=n_latent)
        self.mu_y = nn.Parameter(torch.randn(n_clusters, n_latent))
        self.logvar_y = nn.Parameter(torch.zeros(n_clusters, n_latent)) 
        self.decoderztox = DecoderZtoX(n_output=n_input, n_latent=n_latent, likelihood=likelihood)
        self.n_clusters = n_clusters
        self.n_latent = n_latent


    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"x": tensors[REGISTRY_KEYS.X_KEY]}

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_ = torch.log1p(x) 
        probs_y = self.encoderxtoy(x_)
        y_one_hot = torch.eye(self.n_clusters, device=x.device).unsqueeze(0).repeat(x_.size(0), 1, 1)  # (batch_size, n_clusters, n_clusters)
        x_expanded = x_.unsqueeze(1).repeat(1, self.n_clusters, 1)  # (batch_size, n_clusters, n_input)
        mean_n, logvar_n = self.encoderxytoz(
            x=x_expanded.view(-1, x_.size(-1)),  # Fusion des dimensions pour le traitement batch
            y=y_one_hot.view(-1, self.n_clusters),  # Idem pour y
        )
        mean_n = mean_n.view(x_.size(0), self.n_clusters, -1)  # (batch_size, n_clusters, n_latent)
        logvar_n = logvar_n.view(x_.size(0), self.n_clusters, -1)  # (batch_size, n_clusters, n_latent)
        var_n = logvar_n.exp()
        z_normales = Normal(mean_n, torch.sqrt(var_n)).rsample()  # (batch_size, n_clusters, n_latent)

        return {
            "qzm": mean_n,
            "qzv": var_n,
            "z": z_normales,
            "probs_y": probs_y,
            "z_normales": z_normales,
        }

    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "z_normales": inference_outputs["z_normales"],
        }

    @auto_move_data
    def generative(self, z_normales: torch.Tensor) -> dict[str, torch.Tensor]:
        z_flat = z_normales.view(-1, z_normales.size(-1))  # (batch_size * n_clusters, n_latent)
        generative_outputs = self.decoderztox(z_flat)
        if self.likelihood in ['nb', 'zinb']:
            nb_mean = generative_outputs["mean"].view(z_normales.size(0), z_normales.size(1), -1)
            nb_disp = generative_outputs["disp"].view(z_normales.size(0), z_normales.size(1), -1)
            if self.likelihood == 'zinb':
                zero_prob = generative_outputs["zero_prob"].view(z_normales.size(0), z_normales.size(1), -1)
                return {
                    "mean": nb_mean,
                    "disp": nb_disp,
                    "zero_prob": zero_prob,
                }
            else:
                return {
                    "mean": nb_mean,
                    "disp": nb_disp,
                }
        elif self.likelihood in ['p', 'zip']:
            lambda_ = generative_outputs["lambda"].view(z_normales.size(0), z_normales.size(1), -1)
            if self.likelihood == 'zip':
                zero_prob = generative_outputs["zero_prob"].view(z_normales.size(0), z_normales.size(1), -1)
                return {
                    "lambda": lambda_,
                    "zero_prob": zero_prob,
                }
            else:
                return {
                    "lambda": lambda_,
                }
        else:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")


    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ) -> LossOutput:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(1)

        if self.likelihood == 'nb':
            nb_mean = generative_outputs["mean"]  # (batch_size, n_clusters, n_output)
            nb_disp = generative_outputs["disp"]  # (batch_size, n_clusters, n_output)
            log_likelihood = NegativeBinomial(total_count=nb_disp, logits=torch.log(nb_mean + 1e-4)).log_prob(x_expanded).sum(dim=-1)  # (batch_size, n_clusters)
        elif self.likelihood == 'zinb':
            nb_mean = generative_outputs["mean"]
            nb_disp = generative_outputs["disp"]
            zero_prob = generative_outputs["zero_prob"]
            pi = torch.sigmoid(zero_prob)  # Probabilité d'inflation zéro
            
            nb_dist = NegativeBinomial(total_count=nb_disp, logits=torch.log(nb_mean + 1e-4))
            log_nb_prob = nb_dist.log_prob(x_expanded)
            log_nb_prob_zero = nb_dist.log_prob(torch.zeros_like(x_expanded))
            
            log_likelihood = torch.where(
                x_expanded == 0,
                torch.log(pi + (1 - pi) * torch.exp(log_nb_prob_zero) + 1e-10),
                torch.log(1 - pi) + log_nb_prob
            ).sum(dim=-1)  # (batch_size, n_clusters)
        
        elif self.likelihood == 'p':
            lambda_ = generative_outputs["lambda"]
            poisson_dist = torch.distributions.Poisson(rate=lambda_)
            log_likelihood = poisson_dist.log_prob(x_expanded).sum(dim=-1)  # (batch_size, n_clusters)
        
        elif self.likelihood == 'zip':
            lambda_ = generative_outputs["lambda"]
            zero_prob = generative_outputs["zero_prob"]
            pi = torch.sigmoid(zero_prob)  
            
            poisson_dist = torch.distributions.Poisson(rate=lambda_)
            log_poisson_prob = poisson_dist.log_prob(x_expanded)
            log_poisson_prob_zero = poisson_dist.log_prob(torch.zeros_like(x_expanded))
            
            log_likelihood = torch.where(
                x_expanded == 0,
                torch.log(pi + (1 - pi) * torch.exp(log_poisson_prob_zero) + 1e-10),
                torch.log(1 - pi) + log_poisson_prob
            ).sum(dim=-1)  # (batch_size, n_clusters)
        
        else:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")

        qz_m = inference_outputs["qzm"] # (batch_size, n_clusters, n_latent)
        qz_v = inference_outputs["qzv"] # (batch_size, n_clusters, n_latent)
        probs_y = inference_outputs["probs_y"] # (batch_size, n_clusters)

        mu_y_expanded = self.mu_y.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, n_clusters, n_latent)
        var_y_expanded = self.logvar_y.unsqueeze(0).expand(batch_size, -1, -1).exp()  # (batch_size, n_clusters, n_latent)
        priors_z_y_distributions = Normal(mu_y_expanded, torch.sqrt(var_y_expanded)) # (batch_size, n_clusters, n_latent)
        var_post_dist = Normal(qz_m, torch.sqrt(qz_v)) # (n_batch, n_clusters, n_latent)
        kl_div_1 = kl(var_post_dist, priors_z_y_distributions).sum(dim=-1) # (batch_size, n_clusters)

        avg_cat_ll_kl = ((log_likelihood - kl_div_1) * probs_y).sum(dim=1) # (batch_size)

        q_y_x = torch.distributions.Categorical(probs_y)
        probs_uniform = torch.ones_like(probs_y)/self.n_clusters
        unif_pi = torch.distributions.Categorical(probs_uniform)
        kl_div_2 = kl(q_y_x, unif_pi) # (batch_size)
        elbo = avg_cat_ll_kl - kl_div_2 # (batch_size)
        loss = torch.mean(-elbo)
        return LossOutput(
            loss=loss,
            reconstruction_loss=-avg_cat_ll_kl,
            kl_local = kl_div_2,
        )
    

class GMVAEModel(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """single-cell Variational Auto Encoder [Grønbech18]_."""

    def __init__(
        self,
        adata: AnnData,
        n_clusters: int,
        likelihood: str,
        n_latent: int = 10,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = GMVAEModule(
            n_input=self.summary_stats["n_vars"],
            n_clusters=n_clusters,
            n_latent=n_latent,
            likelihood=likelihood,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"GMVAE Model with the following params: \n"
            f"n_latent: {n_latent}, n_clusters: {n_clusters}, likelihood: {likelihood}"
        )
        self.init_params_ = self._get_init_params(locals())


    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ):

        self._check_if_trained(warn=False)
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )
        latent = {}
        latent_rep = []
        latent_cat = []
        for tensors in dataloader:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs["qzm"]
            probs_y = outputs["probs_y"]
            latent_cat += [probs_y.cpu()]
            probs_y = probs_y.unsqueeze(-1)
            qzm_pond = (qz_m * probs_y).sum(dim=1)
            latent_rep += [qzm_pond.cpu()]
            
        latent["latent_rep"] = torch.cat(latent_rep).detach().numpy()
        latent["latent_cat"] = torch.cat(latent_cat).detach().numpy()
        return latent

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