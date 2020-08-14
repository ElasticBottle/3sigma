from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
from typing import Tuple


class PictureDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self.x, self.y = x, y

    def __getitem__(self, i: int) -> Tuple:
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
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            yield self.idxs[i : i + self.bs]


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss, tot_acc = 0.0, 0.0
            for xb, yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc += accuracy(pred, yb)
        nv = len(valid_dl)
        print(epoch, tot_loss / nv, tot_acc / nv)
    return tot_loss / nv, tot_acc / nv

