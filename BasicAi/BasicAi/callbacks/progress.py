# Much of these code is from https://course.fast.ai/
# Re-typed most of the code as an exercise to better understand the workings
# I claim no credit for any of this
import time
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time

from BasicAi.BasicAi.callbacks import Callback


# class TrainEvalCallback(Callback):
#     def begin_fit(self):
#         "Set the iter and epoch counters to 0, put the model and the right device"
#         self.learner.train_iter, self.learner.pct_train = 0, 0.0
#         if hasattr(self.dls, "device"):
#             self.model.to(self.dls.device)
#         if hasattr(self.model, "reset"):
#             self.model.reset()

#     def begin_train(self):
#         "Set the model in training mode, tracks some info about the training"
#         self.learner.pct_train = self.epoch / self.epochs

#     def after_batch(self):
#         "Update the iter counter (in training mode)"
#         if self.training:
#             self.learner.pct_train += 1.0 / self.n_iter
#             self.learner.train_iter += 1


class Progress(Callback):
    "A `Callback` to handle the display of progress bars and adds ability to print to console any stats"
    # run_after = Recorder
    _order = 10

    def __init__(self, create_mbar: bool = True):
        super().__init__()
        self.create_mbar = create_mbar

    def begin_fit(self):
        assert hasattr(self.learner, "info")
        if self.create_mbar:
            self.mbar = master_bar(range(self.epochs))
        if self.learner.logger != None:
            self.old_logger, self.learner.logger = self.logger, self._logger
            self._logger(self.info.metric_names)
        else:
            self.old_logger = None

    def _logger(self, log):
        if getattr(self, "mbar", False):
            to_write = [
                f"{item:.6f}" if isinstance(item, float) else str(item) for item in log
            ]
            self.mbar.write(to_write, table=True)

    def begin_epoch(self):
        if getattr(self, "mbar", False):
            self.mbar.update(self.epoch)

    def begin_all_batch(self):
        self._launch_pbar()

    def after_all_batch(self):
        self.pbar.on_iter_end()

    def _launch_pbar(self):
        self.pbar = progress_bar(
            self.dl, parent=getattr(self, "mbar", None), leave=False
        )
        self.pbar.update(0)

    def begin_batch(self):
        self.pbar.update(self.iter + 1)

    def after_batch(self):
        if hasattr(self, "ema_loss"):
            self.pbar.comment = f"{self.ema_loss:.4f}"

    def after_fit(self):
        if getattr(self, "mbar", False):
            self.mbar.on_iter_end()
            delattr(self, "mbar")
        if hasattr(self, "old_logger"):
            self.learner.logger = self.old_logger


class ShowGraphCallback(Callback):
    "Update a graph of training and validation loss"
    run_after, run_valid = Progress, False

    def begin_fit(self):
        self.run = not hasattr(self.learner, "lr_finder") and not hasattr(
            self, "gather_preds"
        )
        self.nb_batches = []
        assert hasattr(self.learner, "progress")

    def after_train(self):
        self.nb_batches.append(self.train_iter)

    def after_epoch(self):
        "Plot validation loss in the pbar graph"
        rec = self.learner.recorder
        iters = range_of(rec.losses)
        val_losses = [v[1] for v in rec.values]
        x_bounds = (
            0,
            (self.n_epoch - len(self.nb_batches)) * self.nb_batches[0]
            + len(rec.losses),
        )
        y_bounds = (0, max((max(Tensor(rec.losses)), max(Tensor(val_losses)))))
        self.progress.mbar.update_graph(
            [(iters, rec.losses), (self.nb_batches, val_losses)], x_bounds, y_bounds
        )


#%%
print("hello world")
