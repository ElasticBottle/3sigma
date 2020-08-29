# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this

from typing import Callable, List, Union

import torch

import callbacks as C
from callbacks import Callback, Cb
from data import DataBunch
from training.cancellation import *
from utils import camel_to_snake, make_list


class Learner:
    def __init__(
        self,
        model,
        data: DataBunch,
        loss_func,
        opt_func,
        model_splitter: Callable = lambda x: x.paramters(),
        metrics: Union[List] = [],
        callbacks: Union[Callback, List[Callback]] = [],
    ):
        (
            self.model,
            self.data,
            self.loss_func,
            self.opt_func,
            self.splitter,
            self.metrics,
        ) = (
            model,
            data,
            loss_func,
            opt_func,
            model_splitter,
            metrics,
        )
        self.training, self.logger, self.opt = False, print, None

        self.cbs = []
        self.add_cbs(make_list(callbacks) + C.default_callbacks)

    def add_cbs(self, cbs: Union[Callback, List[Callback]]):
        for cb in make_list(cbs):
            cb.learner = self
            self.cbs.append(cb)
            setattr(self, cb.name, cb)

    def remove_cbs(self, cbs: Union[Callback, List[Callback]]):
        for cb in make_list(cbs):
            cb.learn = None
            if hasattr(self, cb.name):
                delattr(self, cb.name)
            if cb in self.cbs:
                self.cbs.remove(cb)

    def _one_batch(
        self, xb: torch.utils.data.Dataset, yb: torch.utils.data.Dataset,
    ):
        def update_weights():
            self.loss.backward()
            self(Cb.AFTER_BACKWARD)
            self.opt.step()
            self(Cb.AFTER_STEP)
            self.opt.zero_grad()
            self(Cb.AFTER_ZERO_GRAD)

        self.xb, self.yb = xb, yb
        try:
            self.pred = self.model(self.xb)
            self(Cb.AFTER_PRED)
            self.loss = self.loss_func(self.pred, self.yb)
            self(Cb.AFTER_LOSS)
            if self.training:
                update_weights()
        except CancelOneBatchException:
            self(Cb.AFTER_CANCEL_ONE_BATCH)

    def _all_batches(self, data: torch.utils.data.DataLoader):
        try:
            self.dl, self.iters = data, len(data)
            for i, (xb, yb) in enumerate(data):
                self.iter = i
                self(Cb.BEGIN_BATCH)
                self._one_batch(xb, yb)
                self(Cb.AFTER_BATCH)
        except CancelAllBatchException:
            self(Cb.AFTER_CANCEL_ALL_BATCH)

    def _train(self):
        self.training, self.loss = True, torch.zeros(1)
        self(Cb.BEGIN_TRAINING)
        self._all_batches(self.data.train_dl)
        self(Cb.AFTER_TRAINING)

    def _validate(self):
        self.training = False
        with torch.no_grad():
            self(Cb.BEGIN_VALIDATE)
            self._all_batches(self.data.valid_dl)
            self(Cb.AFTER_VALIDATE)

    def fit(
        self,
        epochs,
        lr: float = 1e-5,
        callbacks: Union[Callback, List[Callback]] = None,
        reset_opt=False,
    ):
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        self.add_cbs(callbacks)
        self.epochs = epochs
        try:
            self(Cb.BEGIN_FIT)
            for epoch in range(epochs):
                self.epoch = epoch
                self(Cb.BEGIN_EPOCH)
                self._train()
                self._validate()
                self(Cb.AFTER_EPOCH)

        except CancelFitException:
            self("after_cancel_train")

        self(Cb.AFTER_FIT)
        self.remove_cbs(callbacks)

    def __call__(self, callback_stage: Cb):
        assert callback_stage in Cb
        for cb in sorted(self.cbs, key=lambda x: x._order):
            cb(callback_stage.value)
