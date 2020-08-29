from callbacks.callback import Callback


class TrainEvalCallback(Callback):
    def begin_fit(self):
        "Set the iter and epoch counters to 0, put the model and the right device"
        self.learn.train_iter, self.learn.pct_train = 0, 0.0
        if hasattr(self.dls, "device"):
            self.model.to(self.dls.device)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def begin_train(self):
        "Set the model in training mode, tracks some info about the training"
        self.learn.pct_train = self.epoch / self.epochs
        self.model.train()

    def after_batch(self):
        "Update the iter counter (in training mode)"
        if self.training:
            self.learn.pct_train += 1.0 / self.n_iter
            self.learn.train_iter += 1

    def begin_validate(self):
        "Set the model in validation mode"
        self.model.eval()
