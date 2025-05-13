# Build based on the original code from Lightning AI
# litgpt/packed_dataset.py

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import random
import hashlib
from torch.utils.data import IterableDataset, get_worker_info

# We will build v0 assuming that the dataset is already saved to disk
# in standard hf format. This leaves room for preproc ops as separate logic.
# basic assumpution will be "text" field only.
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


class HuggingfaceDataset(IterableDataset):
    def __init__(
        self,
        ds_name_or_path=None,
        seed=12345,
        shuffle=False,
        num_processes=1,
        process_rank=0,
        data_id=None,
        data_signature: dict[str, list[str] | str] = {"keys": ["text"], "format_fn": "pass_text"},
        repetitions=None,
        return_data_id=False,
    ):
        assert ds_name_or_path is not None
        self._ds_name_or_path = ds_name_or_path
        self._seed = seed
        assert not shuffle, "Shuffle not implemented for hfds."
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._data_id = data_id  # This is human readble, the mixture unit
        self._return_data_id = return_data_id
        self._ds_fingerprint = (
            None  # This is not human readable, corresp to the subset of work _this_ process is handling.
        )
        self._data_signature = data_signature
        self._ds_total_length = None
        self._ds_length = None
        self._subds = None
        self._ds_min = None
        self._ds_max = None

        # Here is where we load the dataset from disk (whole thing, but just the memmap ofc)
        if repetitions is not None:
            ds_list = [load_from_disk(ds_name_or_path) for _ in range(repetitions)]
            self._ds: Dataset = concatenate_datasets(ds_list)  # type: ignore
        else:
            self._ds: Dataset = load_from_disk(ds_name_or_path)  # type: ignore

        assert not isinstance(
            self._ds, DatasetDict
        ), "Dataset path should point to a single split, try adding /train ?."

        self._ds_total_length = len(self._ds)

    def __iter__(self):  # type: ignore
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        # This is where we shard the dataset into work for each dataparallel rank.
        # Our unit of work is now a "row" of the dataset though, not a file.

        self._worker_id = worker_id

        # max_num_rows = (len(self._ds) // num_shards) * num_shards
        max_num_rows = len(self._ds)
        index_list = list(range(shard_id, max_num_rows, num_shards))

        if index_list == []:
            self._ds_fingerprint = None
            self._ds_min = 0
            self._ds_max = 0
        else:
            self._ds_fingerprint = hashlib.shake_128(str(index_list).encode()).hexdigest(4)
            self._ds_min = min(index_list)
            self._ds_max = max(index_list)

        subds = self._ds.select(index_list)
        self._subds = subds

        self._ds_length = len(self._subds)

        logger.info(
            f"Rank {self._process_rank}/{self._num_processes}, worker {worker_id} has "
            f"{self._ds_length}/{self._ds_total_length} rows | identifier={self._data_id}:{self._ds_fingerprint} "
            f"| range={self._ds_min}:{self._ds_max} | head={index_list[:3]} | tail={index_list[-3:]}"
        )

        return HuggingfaceDatasetIterator(
            ds=subds,
            data_signature=self._data_signature,
            data_id=self._data_id,
            return_data_id=self._return_data_id,
            fingerprint=self._ds_fingerprint,
            worker_id=worker_id,
            process_rank=self._process_rank,
            num_processes=self._num_processes,
        )

    def __len__(self):
        return self._ds_length


class HuggingfaceDatasetIterator:
    def __init__(
        self,
        ds,
        data_signature: dict[str, list[str] | str],
        data_id=None,
        return_data_id=None,
        fingerprint=None,
        worker_id=None,
        process_rank=None,
        num_processes=None,
    ):
        self._ds = ds
        self._data_signature = data_signature
        self._data_id = data_id
        self._return_data_id = return_data_id
        self._ds_fingerprint = fingerprint
        self._worker_id = worker_id
        self._process_rank = process_rank
        self._num_processes = num_processes

        self._ds_iter = None

    def __len__(self):
        return len(self._ds)

    def __next__(self):
        if self._ds_iter is None:
            self._ds_iter = iter(self._ds)

        row = next(self._ds_iter)

        # the data signature tells us what keys to extract from the row
        row = {k: row[k] for k in self._data_signature["keys"]}
        # then we attach the data_signature to the sample to support
        # heterogeneously sourced batches in the collate_fn
        row["data_signature"] = self._data_signature

        if self._return_data_id:
            row["data_id"] = self._data_id

        return row


class HuggingfaceCombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None, data_telemetry=False):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        self._data_telemetry = data_telemetry
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets
        else:
            self._weights = [w / sum(weights) for w in weights]

    def __iter__(self):
        return HuggingfaceCombinedDatasetIterator(self._datasets, self._seed, self._weights, self._data_telemetry)


class HuggingfaceCombinedDatasetIterator:
    def __init__(self, datasets, seed, weights, data_telemetry=False):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        self._iter_ct = 0
        self._data_telemetry = data_telemetry

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        self._iter_ct += 1

        # this is the very beginning of data telemetry
        if self._data_telemetry and self._iter_ct < 5:
            logger.info(
                f"Draw result i={self._iter_ct} for rank={dataset._process_rank}/{dataset._num_processes}, "
                f"worker={dataset._worker_id} | {dataset._data_id}:{dataset._ds_fingerprint}"
            )
        elif self._data_telemetry and self._iter_ct == 5:
            logger.info("Data telemetry off ...")

        return next(dataset)
