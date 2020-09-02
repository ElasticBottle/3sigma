# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
from typing import Tuple

import torch


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self.x, self.y = x, y

    def __getitem__(self, i) -> Tuple:
        return (self.x[i], self.y[i])

    def __len__(self) -> int:
        return len(self.x)


def collate(b):
    xs, ys = zip(*b)
    print(xs)
    return torch.stack(xs), torch.stack(ys)


class Sampler:
    def __init__(self, ds, bs, shuffle=False):
        self.n, self.bs, self.shuffle = len(ds), bs, shuffle

    def __iter__(self):
        self.indices = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            yield self.indices[i : i + self.bs]


class DataBunch:
    def __init__(
        self,
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        c: int = None,
    ):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c

    @property
    def train_dl(self):
        return self.__train_dl

    @train_dl.setter
    def train_dl(self, value):
        self.__train_dl = value

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

    @property
    def valid_dl(self):
        return self.__valid_dl

    @valid_dl.setter
    def valid_dl(self, value):
        self.__valid_dl = value
