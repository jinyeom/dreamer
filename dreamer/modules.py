from typing import Optional, Tuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions.one_hot_categorical import OneHotCategoricalStraightThrough


class MLP(nn.Sequential):
    """
    Multilayer perceptron with exponential linear unit activations.

    Parameters:
        in_dim (int): Number of input features
        out_dim (int): Number of outputs
        hid_dims (int): An optional sequence of numbers of hidden units
    """

    def __init__(
        self, in_dim: int, out_dim: int, hid_dims: Optional[Sequence[int]] = None
    ) -> None:
        if hid_dims is None or len(hid_dims) == 0:
            net = [nn.Linear(in_dim, out_dim)]
        else:
            net = [nn.Linear(in_dim, hid_dims[0]), nn.ELU(inplace=True)]
            for in_features, out_features in zip(hid_dims[:-1], hid_dims[1:]):
                net.append(nn.Linear(in_features, out_features))
                net.append(nn.ELU(inplace=True))
            net.append(nn.Linear(hid_dims[-1], out_dim))
        super().__init__(*net)


class Encoder(nn.Module):
    """
    Convolutional encoder for visual observation data. Note that the encoder expects the
    input shape of (B, 3, 64, 64).

    Parameters:
        x_dim (int): Size of the observation embedding
    """

    def __init__(self, x_dim: int) -> None:
        super().__init__()
        self.x_dim = x_dim

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.embed = nn.Linear(1024, self.x_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.elu_(self.conv1(x))
        x = F.elu_(self.conv2(x))
        x = F.elu_(self.conv3(x))
        x = F.elu_(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.embed(x)
        return x


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) is composed of three models: 1) a recurrent
    model, which updates a deterministic recurrent state, 2) a representation model,
    which computes a posterior stochastic state, and 3) a transition model, which
    computes a prior stochastic state that tries to predict the posterior without
    access to the current observation.

    Parameters:
        x_dim (int): Size of the observation embedding
        h_dim (int): Size of the deterministic model state
        z_num_var (int): Number of categorical variables for the stochastic state
        z_var_dim (int): Size of each categorical variable for the stochastic state
        a_dim (int): Size of the action vector
    """

    def __init__(
        self, x_dim: int, h_dim: int, z_num_var: int, z_var_dim: int, a_dim: int
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_num_var = z_num_var
        self.z_var_dim = z_var_dim
        self.z_dim = z_num_var * z_var_dim
        self.a_dim = a_dim

        self.recurrent = nn.GRUCell(self.z_dim + self.a_dim, self.h_dim)
        self.representation = MLP(self.x_dim + self.h_dim, self.z_dim, (1024, 1024))
        self.transition = MLP(self.h_dim, self.z_dim, (1024, 1024))

    def compute_posterior(self, x: Tensor, h: Tensor) -> Tensor:
        z_logits = self.representation(torch.cat([x, h], dim=1))
        z_logits = z_logits.reshape(-1, self.z_num_var, self.z_var_dim)
        z_dist = OneHotCategoricalStraightThrough(logits=z_logits)
        z = z_dist.rsample()
        return z

    def compute_prior(self, h: Tensor) -> Tensor:
        z_hat_logits = self.transition(h)
        z_hat_logits = z_hat_logits.reshape(-1, self.z_num_var, self.z_var_dim)
        z_hat_dist = OneHotCategoricalStraightThrough(logits=z_hat_logits)
        z_hat = z_hat_dist.rsample()
        return z_hat

    def step(
        self, action: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if states is None:
            states = (
                torch.zeros(action.size(0), self.h_dim).to(action.device),
                torch.zeros(action.size(0), self.z_dim).to(action.device),
            )
        h, z = states
        h = self.recurrent(torch.cat([z, action], dim=1), h)
        z = self.compute_prior(h)
        return h, z

    def forward(
        self, x: Tensor, action: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if states is None:
            states = (
                torch.zeros(x.size(0), self.h_dim).to(x.device),
                torch.zeros(x.size(0), self.z_dim).to(x.device),
            )
        h, z = states
        h = self.recurrent(torch.cat([z, action], dim=1), h)
        z = self.compute_posterior(x, h)
        z_hat = self.compute_prior(h)
        return h, z, z_hat


class Decoder(nn.Module):
    """
    Deconvolutional decoder for reconstructing visual observations from model states.

    Parameters:
        h_dim (int): Size of the deterministic model state
        z_dim (int): Size of the stochstic model state
    """

    def __init__(self, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim

        self.fc = nn.Linear(h_dim + z_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        x = torch.cat([h, z], dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = F.elu_(self.fc(x))
        x = F.elu_(self.deconv1(x))
        x = F.elu_(self.deconv2(x))
        x = F.elu_(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x
