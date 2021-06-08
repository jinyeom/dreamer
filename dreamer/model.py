from typing import Tuple, Optional, Sequence

import torch
from torch import Tensor
import pytorch_lightning as pl

from .modules import Encoder, RSSM, Decoder


class WorldModel(pl.LightningModule):
    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        z_num_var: int,
        z_var_dim: int,
        a_dim: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_num_var = z_num_var
        self.z_var_dim = z_var_dim
        self.z_dim = z_num_var * z_var_dim
        self.a_dim = a_dim

        self.encoder = Encoder(self.x_dim)
        self.rssm = RSSM(
            self.x_dim, self.h_dim, self.z_num_var, self.z_var_dim, self.a_dim
        )
        self.decoder = Decoder(self.h_dim, self.z_dim)

    def forward(
        self, action: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        states = self.rssm.step(action, states=states)
        obs = self.decoder(torch.cat(states, dims=1))
        return states, obs

    def training_step(self, batch: Sequence[Tensor], batch_idx: int) -> Tensor:
        recon_loss = 0
        latent_loss = 0

        states = None
        for obs, action, next_obs in zip(batch):
            obs = self.encoder(obs)
            h, z, z_hat = self.rssm(obs, action, states)
            x_hat = self.decoder(h, z)
            states = (h, z)

            recon_loss += self.recon_loss(x_hat, next_obs)
            latent_loss += self.latent_loss(z_hat, z)

        loss = self.recon_coef * recon_loss + self.latent_coef * latent_loss

        self.log("recon_loss", recon_loss)
        self.log("latent_loss", latent_loss)
        self.log("training_loss", loss)

        return loss
