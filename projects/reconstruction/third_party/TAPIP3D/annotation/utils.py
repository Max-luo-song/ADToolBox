import fcntl
import os
import h5py
from pathlib import Path
from typing import Dict, Iterator, Optional, TypeVar, List
from hydra import compose, initialize_config_dir
from tqdm import tqdm
from omegaconf import open_dict
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import Sampler
from concurrent.futures import ThreadPoolExecutor

from datasets.datatypes import RawSliceData, SliceData
from annotation.base_annotator import BaseAnnotator
CONFIG_PATH = Path(__file__).parent / 'provider_configs'

T_co = TypeVar('T_co', covariant=True)

class UnevenDistributedSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, skip_idxs: Optional[List[int]] = None) -> None:
        if num_replicas is None:
            if not torch.dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if skip_idxs is not None:
            self.skip_idxs = set(skip_idxs)
        else:
            self.skip_idxs = set()

        self.num_samples = len(self.dataset) // self.num_replicas
        if len(self.dataset) % self.num_replicas != 0:
            if self.rank % self.num_replicas <= (len(self.dataset) - 1) % self.num_replicas:
                self.num_samples += 1

        self.shuffle = shuffle
        self.seed = seed

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # subsample
        indices = indices[self.rank % self.num_replicas::self.num_replicas]

        self.indices = [idx for idx in indices if idx not in self.skip_idxs]
        self.num_samples = len(self.indices)

    def __iter__(self) -> Iterator[T_co]:
        assert len(self.indices) == self.num_samples
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def load_annotations(
    annotation_path: str,
    key: str,
    seq_id: int, 
    start: int, 
    length: int, 
    stride: int,
) -> np.ndarray:
    assert os.path.exists(os.path.join(annotation_path, f"{seq_id}.confirm")), f"Confirm file {os.path.join(annotation_path, f'{seq_id}.confirm')} does not exist"
    with h5py.File(os.path.join(annotation_path, f"{seq_id}.h5"), 'r') as f:
        ret: np.ndarray = f[key][start:start+length*stride:stride] # type: ignore
    return ret # type: ignore

class SeqDataset(Dataset):
    def __init__(self, provider, num_threads):
        self.provider = provider
        self.num_threads = num_threads
        self.seq_lens = provider.load_seq_lens()

    def __len__(self):
        return len(self.seq_lens)
    
    def __getitem__(self, idx):
        seq_len = self.seq_lens[idx]
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            data: RawSliceData = self.provider.load_slice(
                idx, start=0, length=seq_len, stride=1, rng=np.random.default_rng(seed=0), executor=executor
            ) # type: ignore
        return idx, data

def dummy_collate(batch):
    return batch[0]

def generate_annotations(
    output_path: str,
    annotator: BaseAnnotator,
    provider_cfg: str,
    num_dataloader_threads: int = 4,
    num_dataloader_workers: int = 8,
    mod10: Optional[int] = None,
    _idx: Optional[int] = None,
):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        device = f'cuda:{local_rank % torch.cuda.device_count()}'
    else:
        device = 'cpu'
    annotator.to(device)

    # Load provider config
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH.absolute())):
        cfg = compose(config_name=provider_cfg) # type: ignore
    name = cfg.name
    override_anno = cfg.override_anno
    with open_dict(cfg):
        del cfg['name']
        del cfg['override_anno']

    from datasets.providers.base_provider import BaseDataProvider
    provider = BaseDataProvider.from_config(cfg, name=name, override_anno=override_anno)

    output_dir_path = Path(output_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    dataset = SeqDataset(provider, num_dataloader_threads)

    skip_idxs = []
    for seq_id in range(len(dataset)):
        confirm_path = output_dir_path / f"{seq_id}.confirm"
        if confirm_path.exists():
            skip_idxs.append(seq_id)

    if world_size > 1:
        sampler = UnevenDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, skip_idxs=skip_idxs)
    else:
        sampler = UnevenDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, skip_idxs=skip_idxs)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_dataloader_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=dummy_collate
    )
    if rank == 0:
        dataloader = tqdm(dataloader)

    if _idx is not None:
        assert world_size == 1
        dataloader = [dataset[_idx]]

    for dummy_batch in dataloader:
        seq_id, data = dummy_batch
        if mod10 is not None and (int (seq_id)) % 10 != mod10:
            continue

        seq_output_path = output_dir_path / f"{seq_id}.h5"
        confirm_path = output_dir_path / f"{seq_id}.confirm"

        if confirm_path.exists():
            continue

        annotation: Dict[str, np.ndarray] = annotator(data)

        with open(seq_output_path, 'w+b') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                with h5py.File(lock_file, 'w') as f:
                    for key, value in annotation.items():
                        assert isinstance(value, np.ndarray), "Annotation must be numpy arrays"
                        assert key in data.__dict__ or f'gt_{key}' in data.__dict__ or key == 'depths', f"Key {key} not in data"
                        f.create_dataset(key, shape=value.shape, dtype=value.dtype, compression="gzip")
                        f[key][:] = value  # type: ignore
                    f.flush()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            
        confirm_path.touch()