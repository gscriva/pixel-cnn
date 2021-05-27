from typing import Any, Dict, Sequence, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer
from tqdm import trange

from common.utils import PROJECT_ROOT, nll_loss


class ResBlock(nn.Module):
    def __init__(self, block) -> None:
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        # remove mask_type kwargs
        self.mask_type = kwargs.pop("mask_type")
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer("mask", torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + 1 :] = 0
        # type_mask=A excludes the central pixel
        if self.mask_type == "A":
            self.mask[kh // 2, kw // 2] = 0
        self.mask[kh // 2 + 1 :] = 0
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return F.conv2d(
            x,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self):
        return super(
            MaskedConv2d, self
        ).extra_repr() + ", mask_type={mask_type}".format(**self.__dict__)


class PixelCNN(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        # Force the first x_hat to be 0.5
        if self.hparams.bias:
            self.register_buffer("x_hat_mask", torch.ones([self.hparams.physics.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer(
                "x_hat_bias", torch.zeros([self.hparams.physics.L] * 2)
            )
            self.x_hat_bias[0, 0] = 0.5

        layers = []
        layers.append(
            MaskedConv2d(
                1,
                1 if self.hparams.net_depth == 1 else self.hparams.net_width,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="A",
            )
        )
        for _ in range(self.hparams.net_depth - 2):
            if self.hparams.res_block:
                layers.append(
                    self._build_res_block(
                        self.hparams.net_width, self.hparams.net_width,
                    )
                )
            else:
                layers.append(
                    self._build_simple_block(
                        self.hparams.net_width, self.hparams.net_width,
                    )
                )
        if self.hparams.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.hparams.net_width,
                    self.hparams.net_width if self.hparams.final_conv else 1,
                )
            )
        if self.hparams.final_conv:
            layers.append(
                self._get_activation_func(
                    self.hparams.activation, self.hparams.net_width
                )
            )
            layers.append(nn.Conv2d(self.hparams.net_width, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _get_activation_func(self, activation: str, in_channels: int) -> nn.Module:
        """Returns the requested activation function.

        Args:
            activation (str): Name of the activation function.
            in_channels (int): Number of input channels.

        Returns:
            nn.Module: The chosen activation function.
        """
        if activation == "relu":
            activation_layer = nn.ReLU(in_channels)
        elif activation == "prelu":
            activation_layer = nn.PReLU(in_channels, init=0.5)
        elif activation == "rrelu":
            activation_layer = nn.RReLU(in_channels)
        elif activation == "leakyrelu":
            activation_layer = nn.LeakyReLU(in_channels)
        elif activation == "gelu":
            activation_layer = nn.GELU(in_channels)
        elif activation == "selu":
            activation_layer = nn.SELU(in_channels)
        elif activation == "silu":
            activation_layer = nn.SiLU(in_channels)
        else:
            raise NotImplementedError(
                f"Activation function {activation} not implemented"
            )
        return activation_layer

    def _build_simple_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a masked convolutional block

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Module: Convolutional layer + activation function.
        """
        layers = []
        layers.append(self._get_activation_func(self.hparams.activation, in_channels))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="B",
            )
        )
        return nn.Sequential(*layers)

    def _build_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a convolutional residual block, with a simple conv2d, 
        an activation function and a masked convolutional layer. 

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Module: Residual convolutional block.
        """
        layers = []
        layers.append(self._get_activation_func(self.hparams.activation, in_channels))
        layers.append(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=self.hparams.bias)
        )
        layers.append(
            self._get_activation_func(self.hparams.activation, in_channels // 2)
        )
        layers.append(
            MaskedConv2d(
                in_channels // 2,
                in_channels // 2,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="B",
            )
        )
        layers.append(
            self._get_activation_func(self.hparams.activation, in_channels // 2)
        )
        layers.append(
            nn.Conv2d(in_channels // 2, out_channels, 1, bias=self.hparams.bias)
        )
        return ResBlock(nn.Sequential(*layers))

    def log_prob(self, sample: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Method to compute the logarithm of the probabilty of sample
            and its conditional probaility.

        Args:
            sample (torch.Tensor): Ising Glass sample.
            x_hat (torch.Tensor): Conditional probability of each spin in the sample.

        Returns:
            torch.Tensor: Logarithm of the probabilty of the sample.
        """
        sample, x_hat = sample.double(), x_hat.double()
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.hparams.physics.epsilon) * mask + torch.log(
            1 - x_hat + self.hparams.physics.epsilon
        ) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def _forward(self, x) -> torch.Tensor:
        """Method for the forward pass.

        Returns:
            torch.Tensor: prediction.
        """
        x_hat = self.net(x)

        # Force the first x_hat to be 0.5
        if self.hparams.bias:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias
        return x_hat

    def forward(self, num_sample: int) -> Dict[str, np.ndarray]:
        """Method for generating new sample.

        Args:
            num_sample (int): Sample of Ising Glasses to generate.

        Returns:
            Dict[str, torch.Tensor]: New sample and their probabilities.
        """
        sample = torch.zeros(
            [num_sample, 1, self.hparams.physics.L, self.hparams.physics.L],
            device=self.device,
        )

        for i in trange(self.hparams.physics.L):
            for j in trange(self.hparams.physics.L, leave=False):
                x_hat = self._forward(sample).detach()
                sample[:, :, i, j] = torch.bernoulli(x_hat[:, :, i, j]) * 2 - 1
        # compute probability of the sample
        prob = self.log_prob(sample, x_hat).exp()
        return {"sample": sample.squeeze(1).numpy(), "prob": prob.numpy()}

    def step(self, x) -> torch.Tensor:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the loss.

        Returns:
            torch.Tensor: prediction.
        """
        x_hat = self.net(x)

        # Force the first x_hat to be 0.5
        if self.hparams.bias:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        # compute log_prob of the input
        log_prob = self.log_prob(x, x_hat)

        # compute custom negative log likelihood
        loss = nll_loss(log_prob)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict(
            {"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict(
            {"val_loss": loss}, on_step=False, on_epoch=True, prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict({"test_loss": loss},)
        return loss

    def configure_optimizers(self,) -> Tuple[Sequence[Optimizer], Sequence[Any]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Return:
            Tuple: The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict). 
                    May be present only one optimizer and one scheduler.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        physics=cfg.physics,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
