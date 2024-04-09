import copy
from typing import List, Any, Tuple, Optional, Dict
from er_buffer import ExperienceReplayBuffer, ExperienceReplayBufferState, ExperienceReplayBatchInfo
from sc_depth_data_module import SCDepthDataModule
from pytorch_lightning import LightningModule, Trainer


def initialize_er_buffer(
        hparams: Any, datasets: List[Tuple[str, str, str]],
        global_er_buffer: ExperienceReplayBuffer,
        local_er_buffer_by_participant: Dict[str, ExperienceReplayBuffer],
):
    global_er_buffer_state = global_er_buffer.get_buffer_state()
    local_er_buffer_state_by_participant = {
        participant_id: er_buffer.get_buffer_state()
        for participant_id, er_buffer in local_er_buffer_by_participant.items()
    }

    for dataset_name, dataset_dir, dataset_split in datasets:
        sc_depth_params = copy.deepcopy(hparams)
        sc_depth_params.dataset_name = dataset_name
        sc_depth_params.dataset_dir = dataset_dir
        data_module = SCDepthDataModule(sc_depth_params)
        data_module.setup(dataset_split)
        dataset_size = data_module.get_dataset_size(dataset_split)
        sc_depth_params.epoch_size = int(dataset_size / sc_depth_params.batch_size)
        print(f"Epoch Size: {sc_depth_params.epoch_size} = {dataset_size} / {sc_depth_params.batch_size}")

        if global_er_buffer.is_empty():
            print(f"Initializing ER Buffers with Dataset {dataset_name} of size {dataset_size}")
            er_initializer = ExperienceReplayBufferInitializer(
                dataset_name, global_er_buffer_state, local_er_buffer_state_by_participant
            )
        else:
            print(f"ReInitializing ER Buffer with Dataset {dataset_name} of size {dataset_size}")
            er_initializer = ExperienceReplayBufferReinitializer(
                dataset_name, global_er_buffer_state, local_er_buffer_state_by_participant
            )

        trainer = Trainer(max_epochs=1, min_steps=sc_depth_params.epoch_size)
        if dataset_split == 'train':
            trainer.fit(model=er_initializer, datamodule=data_module)
        else:
            trainer.test(model=er_initializer, datamodule=data_module)

        global_er_buffer_state, local_er_buffer_state_by_participant = er_initializer.get_er_buffers_states()

    global_er_buffer.set_buffer_state(global_er_buffer_state)
    for participant_id, er_buffer in local_er_buffer_by_participant.items():
        er_buffer.set_buffer_state(local_er_buffer_state_by_participant[participant_id])


class ExperienceReplayBufferInitializer(LightningModule):

    def __init__(
            self, dataset_name,
            global_er_buffer_state: ExperienceReplayBufferState,
            local_er_buffer_state_by_participant: Dict[str, ExperienceReplayBufferState],
            *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._global_er_buffer = ExperienceReplayBuffer(-1)
        self._global_er_buffer.set_buffer_state(global_er_buffer_state)
        self._local_er_buffer_state_by_participant = local_er_buffer_state_by_participant

    def get_er_buffers_states(self):
        return self._global_er_buffer.get_buffer_state(with_batch_data=True), self._local_er_buffer_state_by_participant

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx)

    # def validation_step(self, batch, batch_idx):
    #    self._common_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx)

    def _common_step(self, batch, batch_idx):
        self._global_er_buffer.add_batch_to_buffer(
            ExperienceReplayBatchInfo(
                dataset_name=self._dataset_name,
                batch_idx=batch_idx,
                batch_data=batch
            )
        )


class ExperienceReplayBufferReinitializer(LightningModule):

    def __init__(
            self, dataset_name,
            global_er_buffer_state: ExperienceReplayBufferState,
            local_er_buffer_state_by_participant: Dict[str, ExperienceReplayBufferState],
            *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)

        self._dataset_name = dataset_name

        self._global_er_buffer_state = global_er_buffer_state
        self._buffered_batches_global: Dict[int, Tuple[int, ExperienceReplayBatchInfo]] = {}
        for index, batch_info in self._global_er_buffer_state.batches.items():
            if batch_info.dataset_name == dataset_name:
                self._buffered_batches_global[batch_info.batch_idx] = (index, batch_info)

        self._local_er_buffer_state_by_participant = local_er_buffer_state_by_participant
        self._buffered_batches_local_by_participant: Dict[str, Dict[int, Tuple[int, ExperienceReplayBatchInfo]]] = {}
        for participant_id, er_buffer_state in self._local_er_buffer_state_by_participant.items():
            buffered_batches = self._buffered_batches_local_by_participant[participant_id] = {}
            for index, batch_info in er_buffer_state.batches.items():
                if batch_info.dataset_name == dataset_name:
                    buffered_batches[batch_info.batch_idx] = (index, batch_info)

    def get_er_buffers_state(self):
        return self._global_er_buffer_state, self._local_er_buffer_state_by_participant

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx)

    # def validation_step(self, batch, batch_idx):
    #    self._common_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx)

    def _common_step(self, batch, batch_idx):
        # global
        index, batch_info = self._buffered_batches_global.get(batch_idx, (None, None))
        if batch_info:
            print(f"Restoring batch {batch_idx} of {self._dataset_name} to global buffer at index {index} of buffer")
            batch_info.batch_data = batch
            self._global_er_buffer_state.batches[index] = batch_info
        # local
        for participant_id, buffered_batches in self._buffered_batches_local_by_participant.items():
            er_buffer_state = self._local_er_buffer_state_by_participant[participant_id]
            index, batch_info = buffered_batches.get(batch_idx, (None, None))
            if batch_info:
                print(f"Restoring batch {batch_idx} of {self._dataset_name} to buffer of participant {participant_id} "
                      f"at index {index} of buffer")
                batch_info.batch_data = batch
                er_buffer_state.batches[index] = batch_info


