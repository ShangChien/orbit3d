from torch.utils.data import Dataset

class BaseWrapperDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def set_epoch(self, epoch, **unused):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

class NestedDictionaryDataset(Dataset):
    def __init__(self, defn):
        super().__init__()
        self.defn = defn
        first = next(iter(defn.values()))
        self._len = len(first)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.defn.items()}

class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed

import contextlib
import numpy as np

class _DataUtils:
    @staticmethod
    @contextlib.contextmanager
    def numpy_seed(seed, *addl_seeds, key=None):
        if seed is None:
            yield
            return
        if len(addl_seeds) > 0:
            seed = int(hash((seed, *addl_seeds)) % 1e8)
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

data_utils = _DataUtils()
