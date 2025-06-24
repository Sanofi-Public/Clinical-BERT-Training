import os
import json

from transformers import TrainerCallback


class DelayedEarlyStoppingCallback(TrainerCallback):
    """
    An early stopping callback that takes effect after a certain number of epochs
    """

    def __init__(self, early_stopping_callback, delay_epochs=0.5):
        self.early_stopping_callback = early_stopping_callback
        self.delay_epochs = delay_epochs

    def on_evaluate(self, args, state, control, **kwargs):
        # Check if the delay period has passed
        if state.epoch >= self.delay_epochs:
            # Trigger early stopping
            self.early_stopping_callback.on_evaluate(args, state, control, **kwargs)


class MetricsCallback(TrainerCallback):
    """
    A training callback for logging evaluations.
    """

    def __init__(self, logging_directory):
        self.training_history = []
        self.logging_directory = logging_directory

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.training_history.append(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        with open(os.path.join(self.logging_directory, "eval_history.json"), "w") as f:
            json.dump(self.training_history, f)


class SaveMetricsCallback(TrainerCallback):
    """
    A training callback for logging training metrics and eval metrics in real time.
    """

    def __init__(self, logging_directory):
        self.logging_directory = logging_directory
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_metrics = {"step": state.global_step, **logs}
        self.metrics_history.append(current_metrics)

        # Save the metrics history to a file
        with open(
            os.path.join(self.logging_directory, "training_metrics.json"), "w"
        ) as f:
            json.dump(self.metrics_history, f)

    def on_train_end(self, args, state, control, **kwargs):
        with open(
            os.path.join(self.logging_directory, "training_metrics.json"), "w"
        ) as f:
            json.dump(self.metrics_history, f)
