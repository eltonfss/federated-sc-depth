import copy
from typing import List, Any, Tuple, Optional, Dict
from er_buffer import ExperienceReplayBuffer, ExperienceReplayBufferState, ExperienceReplayBatchInfo
from sc_depth_data_module import SCDepthDataModule
from pytorch_lightning import LightningModule, Trainer


def initialize_er_buffer(
        hparams: Any, datasets: List[Tuple[str, str]],
        er_buffer: ExperienceReplayBuffer, er_buffer_state: Optional[ExperienceReplayBufferState] = None
):
    for dataset_name, dataset_dir in datasets:
        sc_depth_params = copy.deepcopy(hparams)
        sc_depth_params.dataset_name = dataset_name
        sc_depth_params.dataset_dir = dataset_dir
        data_module = SCDepthDataModule(sc_depth_params)
        data_module.setup("train")
        dataset_size = data_module.get_dataset_size("train")
        print(f"Initializing ER Buffer with Dataset {dataset_name} of size {dataset_size}")
        sc_depth_params.epoch_size = int(dataset_size / sc_depth_params.batch_size)
        print(f"Epoch Size: {sc_depth_params.epoch_size} = {dataset_size} / {sc_depth_params.batch_size}")
        if er_buffer_state:
            er_initializer = ExperienceReplayBufferReinitializer(dataset_name, er_buffer_state)
        else:
            er_initializer = ExperienceReplayBufferInitializer(dataset_name, er_buffer)
        Trainer(max_epochs=1, min_steps=sc_depth_params.epoch_size).fit(model=er_initializer, datamodule=data_module)


class ExperienceReplayBufferInitializer(LightningModule):

    def __init__(self, dataset_name, er_buffer: ExperienceReplayBuffer, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._er_buffer = er_buffer

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._er_buffer.add_batch_to_buffer(
            ExperienceReplayBatchInfo(
                dataset_name=self._dataset_name,
                batch_idx=batch_idx,
                batch_data=batch
            )
        )


class ExperienceReplayBufferReinitializer(LightningModule):

    def __init__(self, dataset_name, er_buffer_state: ExperienceReplayBufferState, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._er_buffer_state = er_buffer_state
        self._buffered_batches: Dict[int, Tuple[int, ExperienceReplayBatchInfo]] = {}
        for index, batch_info in er_buffer_state.batches.items():
            if batch_info.dataset_name == dataset_name:
                self._buffered_batches[batch_info.batch_idx] = (index, batch_info)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        index, batch_info = self._buffered_batches.get(batch_idx, (None, None))
        if batch_info:
            print(f"Restoring batch {batch_idx} of {self._dataset_name} to buffer at index {index} of buffer")
            batch_info.batch_data = batch
            self._er_buffer_state.batches[index] = batch_info
