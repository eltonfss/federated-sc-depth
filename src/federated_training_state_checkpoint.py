from typing import Any

from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from utils import backup_federated_training_state


class BackupFederatedTrainingStateCallback(Callback):

    def __init__(self, model_save_dir, federated_training_state, every_n_train_steps):
        self._model_save_dir = model_save_dir
        self._federated_training_state = federated_training_state
        self._every_n_train_steps = every_n_train_steps
        super().__init__()

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
    ) -> None:
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)
        if not skip_batch:
            backup_federated_training_state(self._model_save_dir, self._federated_training_state)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        backup_federated_training_state(self._model_save_dir, self._federated_training_state)
