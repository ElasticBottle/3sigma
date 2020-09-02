# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
import torch
import re
from enum import Enum
from functools import partial
from BasicAi.BasicAi.utils import camel_to_snake, make_list


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
        - model: nn.Module
        - data: DataBunch
        - opt_func : Callable
        - opt: Optimizer
        - loss_func: Callable
        - Metrics: List
        - Callbacks: List
        - training: bool
        
        BEGIN_TRAINING on:
        - loss: tensor (value only updated AFTER_LOSS)

        AFTER_PRED on:
        - pred: tensor

        BEGIN_BATCH on:
        - dl, xb, yb :DataLoader, DataSet, DataSet
        - epochs, epoch, iters, iter : int for all
    """

    _order = 0

    def __getattr__(self, k):
        return getattr(self.learner, k)

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel_to_snake(name or "callback")

    @property
    def learner(self):
        return self.learner

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f is not None:
            f()

