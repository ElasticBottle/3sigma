from BasicAi.BasicAi.callbacks.callback import Callback, Cb
from BasicAi.BasicAi.callbacks.info import Info
from BasicAi.BasicAi.callbacks.progress import Progress
from BasicAi.BasicAi.callbacks.metrics import Loss, Accuracy, Metric

default_callbacks = [Info(), Progress()]
