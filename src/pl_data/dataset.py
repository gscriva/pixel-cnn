from typing import Dict, Tuple, Union

import hydra
import omegaconf
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset, TensorDataset

from common.utils import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(
        self, name: ValueNode, path: ValueNode, coupling_path: ValueNode, **kwargs
    ):
        super().__init__()
        self.couplings_path = coupling_path
        self.path = path
        self.name = name

        sample = np.load(self.path)
        # couplings have shape 2 x L x L
        couplings = np.load(self.couplings_path)
        couplings = np.repeat(couplings[None, :, :, :], sample.shape[0], axis=0)

        # dataset has now shape N x 3 x L x L
        dataset = np.concatenate((sample[:, None, :, :], couplings), axis=1)

        self.dataset = TensorDataset(torch.from_numpy(dataset).float())

        del couplings, dataset, sample

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # [0] is needed because only one element is returned
        return self.dataset[index][0]

    def __repr__(self) -> str:
        return f"MyDataset(name={self.name}, path={self.path})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
