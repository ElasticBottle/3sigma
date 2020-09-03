# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
import time
import matplotlib.pyplot as plt
import numpy as np

import BasicAi.BasicAi.callbacks as C
from BasicAi.BasicAi.callbacks.callback import Callback
from fastprogress.fastprogress import format_time


class Info(Callback):
    """Callback that tracks statistics (Hyper-parameters, losses and metrics) during training and validation"""

    _order = 1

    def __init__(
        self, add_time=True, train_metrics=False, valid_metrics=True, loss_beta=0.95
    ):
        (
            self.is_add_time,
            self.is_train_metrics,
            self.is_valid_metrics,
            self.loss_beta,
        ) = (add_time, train_metrics, valid_metrics, loss_beta)
        self.loss, self.ema_loss = C.Loss(beta=0), C.Loss(beta=loss_beta)

    def begin_fit(self):
        "Prepare state for training"
        self.lrs, self.iters, self.losses, self.values = [], [], [], []

        self.metric_names = self._get_metric_names()
        self.ema_loss.reset()

    def _get_metric_names(self):
        names = list(map(lambda x: getattr(x, "name"), self.metrics))
        train_names, valid_names = [], []
        if self.is_train_metrics:
            train_names = list(map(lambda x: f"Train_{x}", names))
        if self.is_valid_metrics:
            valid_names = list(map(lambda x: f"Valid_{x}", names))
        names = ["Train_Loss"] + train_names + ["Valid_Loss"] + valid_names
        if self.is_add_time:
            names.append("Time")
        return ["Epoch"] + names

    def begin_epoch(self):
        "Set timer if `self.is_add_time=True`"
        if self.is_add_time:
            self.epoch_start_time = time.time()
        self.log = [self.epoch]

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learner.final_record = self.log[1:].copy()
        self.values.append(self.learner.final_record)
        if self.is_add_time:
            self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.iter)

    def begin_training(self):
        map(lambda x: x.reset(), self._get_metrics())

    def begin_validate(self):
        map(lambda x: x.reset(), self._get_metrics())

    # def after_training(self):
    #     self.log += self._train_mets.map(_maybe_item)
    # def after_validate(self):
    #     self.log += self._valid_mets.map(_maybe_item)

    def after_batch(self):
        "Update all metrics and records lr and ema loss in training"
        if len(self.yb) == 0:
            return
        metrics = self._get_metrics()
        for metric in metrics:
            metric.score(self.pred, self.yb, len(self.yb))
        if not self.training:
            return
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.ema_loss.value)
        self.learn.ema_loss = self.ema_loss.value

    def _get_metrics(self):
        metrics = ([self.ema_loss] if self.training else [self.loss]) + self.metrics
        if (self.training and self.is_train_metrics) or (
            not self.training and self.is_valid_metrics
        ):
            return metrics
        return []

    @property
    def _train_mets(self):
        if getattr(self, "cancel_train", False):
            return L()
        return L(self.ema_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_mets(self):
        if getattr(self, "cancel_valid", False):
            return L()
        return L(self.loss) + self.metrics if self.valid_metrics else L()

    def plot_loss(self, skip_start=5, with_valid=True):
        plt.plot(
            list(range(skip_start, len(self.losses))),
            self.losses[skip_start:],
            label="train",
        )
        if with_valid:
            idx = (np.array(self.iters) < skip_start).sum()
            plt.plot(self.iters[idx:], L(self.values[idx:]).itemgot(1), label="valid")
            plt.legend()


#%%
test = ["1", "2"]
add = False
new = test + (["hello"] if add else ["bye"])
new
