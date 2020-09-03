#%%
# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
import sys
from typing import Callable, Union
from functools import partial
import torch
from BasicAi.BasicAi.callbacks import Callback
from BasicAi.BasicAi.utils import camel_to_snake


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


class Metric:
    def __init__(self, custom_name: str = None):
        super().__init__()
        self.count = 0
        self.total = 0.0
        self.name = custom_name

    def reset(self):
        self.count = 0
        self.total = 0.0

    def score(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs):
        assert len(y_pred) == len(y)
        batch_size = len(y)
        self.count += batch_size
        self.total += self.calculation(y_pred, y, **kwargs) * batch_size
        assert isinstance(self.total, float)

    def calculation(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> float:
        raise NotImplementedError

    @property
    def value(self):
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self):
        return (
            camel_to_snake(self.__class__.__name__)
            if self._name is None
            else self._name
        )

    @name.setter
    def name(self, value: Union[str, None]):
        self._name = value

    def __repr__(self):
        return f"{self.name}: {self.value}"


class Accuracy(Metric):
    """
    Calculates the top_k accuracy for a given classification task
    """

    def calculation(self, y_pred: torch.Tensor, y: torch.Tensor) -> float:
        top_k_results = torch.topk(y_pred, self.top_k)
        indices = 1  # 0 for values
        results = torch.tensor(
            list(map(lambda x: x[1] in top_k_results[indices][x[0]], enumerate(y)))
        )
        return torch.sum(results, dtype=int, axis=0).item() / len(results)

    def __init__(self, top_k: int = 1, custom_name: str = None):
        """
        Args:
            top_k(int): the number of predicted labels to consider when classifying whether the label is correct
        """
        super().__init__(custom_name=custom_name)
        self.top_k = top_k


class Precision(Metric):
    """
    Calculate the percentage of correct predictions over the total number of actual predictions for a particular label
    """

    def __init__(self, positive_class: int, custom_name=None):
        """
        Args:
            - positive_class(int): the class itemized label for which a precision score is to be calculated
        """
        super().__init__(custom_name=custom_name)
        self.positive_class = positive_class

    def calculation(self, y_pred, y) -> float:
        pred_labels = torch.argmax(y_pred.squeeze(), dim=-1)
        pred_positive = torch.tensor(list(map(self._get_positive, pred_labels)))
        total_pred_positive = pred_positive.sum(dim=(0))
        if total_pred_positive == 0:
            return 0.0
        true_positive = torch.tensor(
            list(map(partial(self._get_true_positive, y=y), enumerate(pred_labels)))
        )
        true_positive = true_positive.sum(dim=(0))
        return true_positive.item() / total_pred_positive.item()

    def _get_positive(self, predicted):
        """
        checks to see if the predicted label is the positive class
        """
        return predicted == self.positive_class

    def _get_true_positive(self, index_predicted, y):
        """
        Checks to see if the predicted label is the positive class
        and is also the correct label for that item
        """
        index = index_predicted[0]
        pred = index_predicted[1]
        return pred == y[index] == self.positive_class


class Recall(Metric):
    def __init__(self, positive_class: int, custom_name=None):
        super().__init__(custom_name=custom_name)
        self.positive_class = positive_class

    def calculation(self, y_pred, y) -> float:
        pred_labels = torch.argmax(y_pred.squeeze(), dim=-1)
        actual_positive = torch.tensor(
            list(map(lambda x: x == self.positive_class, pred_labels))
        )
        total_actual_positive = actual_positive.sum(dim=(0))
        if total_actual_positive == 0:
            return 0.0
        true_positive = torch.tensor(
            list(map(partial(self._get_true_positive, y=y), enumerate(pred_labels)))
        )
        true_positive = true_positive.sum(dim=(0))
        return true_positive.item() / total_actual_positive.item()

    def _get_true_positive(self, index_predicted, y):
        """
        Checks to see if the predicted label is the positive class
        and is also the correct label for that item
        """
        index = index_predicted[0]
        pred = index_predicted[1]
        return pred == y[index] == self.positive_class


class Loss(Metric):
    """
    Tracks the loss of a given training loop
    """

    def __init__(self, beta: float = 0, custom_name: str = None):
        """
        Args:
            - beta(float): the weight of old loss, 0 for regular Loss
        """
        # TODO(ElasticBottle): Fix up smooth loss
        super().__init__(custom_name=custom_name)
        self.beta = beta

    def score(self, loss: float, batch_size: int):

        self.count += batch_size
        self.total += loss * batch_size


#%%
import torch
import numpy as np

y_pred = torch.arange(21, dtype=float, requires_grad=True).view(7, 3) / 21
print(y_pred)
y = torch.empty(7, dtype=torch.long).fill_(2.0)
y[0] = 1
y[1] = 0

p = Recall(2)
p.score(y_pred, y)
p.value
