from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Optimizer

from common.utils import PROJECT_ROOT, nll_loss


class ResBlock(nn.Module):
    def __init__(self, block) -> None:
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        self.exclusive = kwargs.pop("exclusive")
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer("mask", torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive) :] = 0
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
        ).extra_repr() + ", exclusive={exclusive}".format(**self.__dict__)


class PixelCNN(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        # Force the first x_hat to be 0.5
        if self.hparams.bias and not self.hparams.pysics.z2:
            self.register_buffer("x_hat_mask", torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer("x_hat_bias", torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 0.5

        layers = []
        layers.append(
            MaskedConv2d(
                1,
                1 if self.hparams.net_depth == 1 else self.hparams.net_width,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                exclusive=True,
            )
        )
        for count in range(self.hparams.net_depth - 2):
            if self.hparams.res_block:
                layers.append(
                    self._build_res_block(
                        self.hparams.net_width, self.hparams.net_width
                    )
                )
            else:
                layers.append(
                    self._build_simple_block(
                        self.hparams.net_width, self.hparams.net_width
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
            layers.append(nn.PReLU(self.hparams.net_width, init=0.5))
            layers.append(nn.Conv2d(self.hparams.net_width, 1, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a masked convolutional block

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Module: Convolutional layer + activation function.
        """
        layers = []
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                exclusive=False,
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
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.bias))
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                exclusive=False,
            )
        )
        return ResBlock(nn.Sequential(*layers))

    def _log_prob(self, sample: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Helper to compute log prob of given sample and its conditional probaility.

        Args:
            sample (torch.Tensor): Ising Glass sample.
            x_hat (torch.Tensor): Conditional probability of each spin in the sample.

        Returns:
            torch.Tensor: Logarithm of the probabilty of the sample.
        """
        mask = (sample + 1) / 2
        log_prob = torch.log(x_hat + self.hparams.physics.epsilon) * mask + torch.log(
            1 - x_hat + self.hparams.physics.epsilon
        ) * (1 - mask)
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
        """Method to compute the log_prob of sample.

        Args:
            sample (torch.Tensor): Sample of Ising Glass.

        Returns:
            torch.Tensor: Log probability of input configuration.
        """
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob

    def _forward(self, x) -> torch.Tensor:
        """
        Method for the forward pass.
        Returns:
            torch.Tensor: prediction.
        """

        x_hat = self.net(x)

        # Force the first x_hat to be 0.5
        if self.hparams.bias and not self.hparams.physics.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass used for generating new sample.
        Returns:
            torch.Tensor: prediction.
        """
        sample = torch.zeros([batch_size, 1, self.L, self.L], device=self.device)
        for i in range(self.hparams.physics.L):
            for j in range(self.hparams.physics.L):
                x_hat = self.step(sample)
                sample[:, :, i, j] = torch.bernoulli(x_hat[:, :, i, j]) * 2 - 1

        if self.z2:
            # Binary random int 0/1
            flip = torch.randint(2, [batch_size, 1, 1, 1], device=self.device) * 2 - 1
            sample *= flip

        # compute log probability of the sample
        log_prob = self.log_prob(sample)

        return {"sample": sample, "prob": log_prob}

    def step(self, x) -> torch.Tensor:
        """Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the loss.
        Returns:
            torch.Tensor: prediction.
        """
        x_hat = self.net(x)

        # Force the first x_hat to be 0.5
        if self.hparams.bias and not self.hparams.physics.z2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        # compute log_prob of the input
        log_prob = self._log_prob(x, x_hat)

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
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
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
