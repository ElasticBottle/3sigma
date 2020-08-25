# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
import torch
import re
from enum import Enum
from functools import partial
import matplotlib.pyplot as plt
from training.cancellation import *
from utils import camel_to_snake, make_list


class Cb(Enum):
    BEGIN_FIT = ("begin_fit",)
    AFTER_FIT = ("after_fit",)
    AFTER_CANCEL_FIT = ("after_cancel_fit",)
    BEGIN_EPOCH = ("begin_epoch",)
    AFTER_EPOCH = ("after_epoch",)
    BEGIN_TRAINING = ("begin_training",)
    AFTER_TRAINING = ("after_training",)
    BEGIN_BATCH = ("begin_batch",)
    AFTER_BATCH = ("after_batch",)
    AFTER_CANCEL_ONE_BATCH = ("after_cancel_one_batch",)
    AFTER_CANCEL_ALL_BATCH = ("after_cancel_all_batch",)
    AFTER_PRED = ("after_pred",)
    AFTER_LOSS = ("after_loss",)
    AFTER_BACKWARD = ("after_backward",)
    AFTER_STEP = ("after_step",)
    AFTER_ZERO_GRAD = ("after_zero_grad",)
    BEGIN_VALIDATE = ("begin_validate",)
    AFTER_VALIDATE = ("after_validate",)


class Callback:
    """ 
    Base class for callbacks

    Possible values to be obtained from Learner:
        Anytime:
        - model
        - data: DataBunch
        - opt_func
        - opt
        - loss_func: Callable
        - Metrics: List
        - Callbacks: List
        - training: bool
        
        BEGIN_TRAINING on:
        - loss: tensor
        AFTER_PRED on:
        - pred: tensor

        BEGIN_BATCH on:
        - dl, xb, yb :DataLoader, DataSet

        BEGIN_BATCH on:
        - epochs, epoch, iters, iter : int
    """

    _order = 0

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel_to_snake(name or "callback")

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f is not None:
            f()


class TrainEvalCallback(Callback):
    def before_fit(self):
        "Set the iter and epoch counters to 0, put the model and the right device"
        self.learn.train_iter, self.learn.pct_train = 0, 0.0
        if hasattr(self.dls, "device"):
            self.model.to(self.dls.device)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def after_batch(self):
        "Update the iter counter (in training mode)"
        self.learn.pct_train += 1.0 / (self.n_iter * self.n_epoch)
        self.learn.train_iter += 1

    def before_train(self):
        "Set the model in training mode"
        self.learn.pct_train = self.epoch / self.n_epoch
        self.model.train()
        self.learn.training = True

    def before_validate(self):
        "Set the model in validation mode"
        self.model.eval()
        self.learn.training = False


class AvgStats:
    def __init__(
        self, metrics, in_train: bool,
    ):
        self.metrics, self.in_train = make_list(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = torch.tensor((0.0)), 0
        self.tot_mets = [0.0] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(
        self,
        loss: torch.Tensor,
        batch_size: int,
        y_batch: torch.utils.data.Dataset,
        predictions: torch.Tensor,
    ):
        self.tot_loss += loss * batch_size
        self.count += batch_size
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(predictions, y_batch) * batch_size


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = (
            AvgStats(metrics, True),
            AvgStats(metrics, False),
        )

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg["lr"])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])

    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[: len(self.losses) - skip_last])

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])


class Recorder(Callback):
    "Callback that registers statistics (lr, loss and metrics) during training"
    remove_on_fetch, run_after = True, TrainEvalCallback

    def __init__(
        self, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98
    ):
        store_attr(self, "add_time,train_metrics,valid_metrics")
        self.loss, self.smooth_loss = AvgLoss(), AvgSmoothLoss(beta=beta)

    def before_fit(self):
        "Prepare state for training"
        self.lrs, self.iters, self.losses, self.values = [], [], [], []
        names = self.metrics.attrgot("name")
        if self.train_metrics and self.valid_metrics:
            names = L("loss") + names
            names = names.map("train_{}") + names.map("valid_{}")
        elif self.valid_metrics:
            names = L("train_loss", "valid_loss") + names
        else:
            names = L("train_loss") + names
        if self.add_time:
            names.append("time")
        self.metric_names = "epoch" + names
        self.smooth_loss.reset()

    def after_batch(self):
        "Update all metrics and records lr and smooth loss in training"
        if len(self.yb) == 0:
            return
        mets = self._train_mets if self.training else self._valid_mets
        for met in mets:
            met.accumulate(self.learn)
        if not self.training:
            return
        self.lrs.append(self.opt.hypers[-1]["lr"])
        self.losses.append(self.smooth_loss.value)
        self.learn.smooth_loss = self.smooth_loss.value

    def before_epoch(self):
        "Set timer if `self.add_time=True`"
        self.cancel_train, self.cancel_valid = False, False
        if self.add_time:
            self.start_epoch = time.time()
        self.log = L(getattr(self, "epoch", 0))

    def before_train(self):
        self._train_mets[1:].map(Self.reset())

    def before_validate(self):
        self._valid_mets.map(Self.reset())

    def after_train(self):
        self.log += self._train_mets.map(_maybe_item)

    def after_validate(self):
        self.log += self._valid_mets.map(_maybe_item)

    def after_cancel_train(self):
        self.cancel_train = True

    def after_cancel_validate(self):
        self.cancel_valid = True

    def after_epoch(self):
        "Store and log the loss/metric values"
        self.learn.final_record = self.log[1:].copy()
        self.values.append(self.learn.final_record)
        if self.add_time:
            self.log.append(format_time(time.time() - self.start_epoch))
        self.logger(self.log)
        self.iters.append(self.smooth_loss.count)

    @property
    def _train_mets(self):
        if getattr(self, "cancel_train", False):
            return L()
        return L(self.smooth_loss) + (self.metrics if self.train_metrics else L())

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


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos):
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):
    return start


@annealer
def sched_exp(start, end, pos):
    return start * (end / start) ** pos


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.0
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        if idx == 2:
            idx = 1
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner


# This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))


class LR_Find(Callback):
    # TODO(eb): save weights and load it back after finishing
    _order = 1

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train:
            return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss * 10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss

