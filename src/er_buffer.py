from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Union, Optional
import numpy as np
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ExperienceReplayBatchInfo:
    dataset_name: str
    batch_idx: int
    batch_data: Optional[Any] = None


@dataclass_json
@dataclass
class ExperienceReplayBufferState:
    size: int
    experience: int
    batches: Dict[int, ExperienceReplayBatchInfo]


ExperienceReplayBatchesReservoir = Dict[int, ExperienceReplayBatchInfo]


class ExperienceReplayBuffer:
    """
    Experience Replay Buffer used to randomly sample past batches using reservoir algorithm.

    ""The first step of any reservoir algorithm is to put the first n
    records of the file into a “reservoir.” The rest of the records are processed
    sequentially; records can be selected for the reservoir only as they are processed.
    An algorithm is a reservoir algorithm if it maintains the invariant that after each
    record is processed a true random sample of size n can be extracted from the
    current state of the reservoir""
    Jeffrey S Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software (TOMS), 1985.
    """

    def __init__(self, buffer_size: int):
        self._reservoir_size = buffer_size
        self._batches_reservoir: ExperienceReplayBatchesReservoir = {}
        self._n_batches_seen = 0

    def __len__(self) -> int:
        return min(self._n_batches_seen, self._reservoir_size)

    def _sample_reservoir_index(self) -> int:
        """
        Sample and index to be filled (or replaced) by the new batch
        """
        if self._n_batches_seen < self._reservoir_size:
            return self._n_batches_seen

        rand = np.random.randint(0, self._n_batches_seen + 1)
        if rand < self._reservoir_size:
            return rand
        else:
            return -1

    def add_batch_to_buffer(self, batch_info: ExperienceReplayBatchInfo):
        """
        Adds the batch info (dataset, batch_idx, batch) to the buffer according to the reservoir strategy.
        """
        reservoir_index = self._sample_reservoir_index()
        #print(f"Adding batch {batch_info.batch_idx} of {batch_info.dataset_name} to buffer at index {reservoir_index}")
        self._n_batches_seen += 1
        if reservoir_index >= 0:
            self._batches_reservoir[reservoir_index] = batch_info

    def get_batches_from_buffer(self, n: int, transform=None) -> List[ExperienceReplayBatchInfo]:
        """
        Randomly samples up to n batches from buffer
        (if buffer has less than n batches, the maximum available is sampled)
        :param n: the number of requested batches
        :param transform: optional transformation to be applied
        :return list of batch info (dataset, batch_idx, batch)
        """

        if transform is None:
            def transform(x: ExperienceReplayBatchInfo) -> ExperienceReplayBatchInfo: return x

        # cap n based on number of batches available
        n_batches_available = min(self._n_batches_seen, len(self._batches_reservoir))
        n = min(n, n_batches_available)

        # randomly chose the sample indexes from reservoir
        batch_indexes = np.random.choice(n_batches_available, size=n, replace=False)
        batches = [transform(batch) for batch in [self._batches_reservoir[index] for index in batch_indexes]]

        return batches

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self._n_batches_seen == 0:
            return True
        else:
            return False

    def get_buffer_state(self, with_batch_data: bool = False) -> ExperienceReplayBufferState:
        return ExperienceReplayBufferState(
            size=self._reservoir_size,
            experience=self._n_batches_seen,
            batches={
                reservoir_index: ExperienceReplayBatchInfo(
                    dataset_name=batch_info.dataset_name,
                    batch_idx=batch_info.batch_idx,
                    batch_data=batch_info.batch_data if with_batch_data else None
                )
                for reservoir_index, batch_info in self._batches_reservoir.items()
            }
        )

    def set_buffer_state(self, buffer_state: ExperienceReplayBufferState):
        self._reservoir_size = buffer_state.size
        self._n_batches_seen = buffer_state.experience
        self._batches_reservoir = buffer_state.batches

