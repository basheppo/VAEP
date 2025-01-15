import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
import csv
import warnings




class CustomReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, monitor="val_loss", optimizer=None, factor=0.95, patience=5, min_lr=1e-6, verbose=0, **kwargs):
        super(CustomReduceLROnPlateau, self).__init__(
            monitor=monitor, factor=factor, patience=patience, min_lr=min_lr, verbose=verbose, **kwargs)
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.optimizer == "vae_optimizer":
            logs["vae_learning_rate"] = float(
                tf.keras.backend.get_value(self.model.vae_optimizer.learning_rate))
        elif self.optimizer == "predictor_optimizer":
            logs["predictor_learning_rate"] = float(
                tf.keras.backend.get_value(self.model.predictor_optimizer.learning_rate))
        else:
            raise ValueError(
                "The Optimizer is not correct ")

        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    if self.optimizer == "vae_optimizer":
                        old_lr = float(tf.keras.backend.get_value(
                            self.model.vae_optimizer.learning_rate))
                        optimizer = self.model.vae_optimizer
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            self.model.vae_optimizer.learning_rate.assign(
                                new_lr)
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch + 1}: "
                                    "ReduceLROnPlateau reducing VAE's "
                                    f"learning rate to {new_lr}."
                                )

                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                    elif self.optimizer == "predictor_optimizer":
                        old_lr = float(tf.keras.backend.get_value(
                            self.model.predictor_optimizer.learning_rate))
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            self.model.predictor_optimizer.learning_rate.assign(
                                new_lr)
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch + 1}: "
                                    "ReduceLROnPlateau reducing Predictor's "
                                    f"learning rate to {new_lr}."
                                )

                        self.cooldown_counter = self.cooldown
                        self.wait = 0




class LogLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):

        lr_vae = tf.keras.backend.get_value(self.model.vae_optimizer.lr)
        lr_pred = tf.keras.backend.get_value(self.model.predictor_optimizer.lr)
        print(
            f"Epoch {epoch + 1}: Learning rate of VAE's optimizer is {lr_vae}.")
        print(
            f"Epoch {epoch + 1}: Learning rate of Predictor's optimizer  {lr_pred}.")


class LogMetricsCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs is not None:
            print(f"Epoch {epoch + 1}:")
            for metric, value in logs.items():
                print(f"{metric}: {value:.4f}")


class CSVLogger(Callback):
    def __init__(self, filename):
        super(CSVLogger, self).__init__()
        self.filename = filename
        self.file_initialized = False

    def on_epoch_end(self, epoch, logs=None):
        if not self.file_initialized:
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers = ['epoch'] + list(logs.keys())
                writer.writerow(headers)
                self.file_initialized = True

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # print([logs[key] for key in logs.keys()])
            row = [epoch] + [logs[key] for key in logs.keys()]
            # print(row)
            writer.writerow(row)