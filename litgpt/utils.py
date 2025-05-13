# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""
import inspect
import math
import pickle
import shutil
import sys
import os

from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    Union,
    Literal,
    Callable,
    Type,
    TypedDict,
    Tuple,
)
from functools import partial
from contextlib import nullcontext, contextmanager

import lightning as L
import torch
import torch.nn as nn
import torch.utils._device
import numpy as np
import random
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning.pytorch.loggers import WandbLogger
from torch.serialization import normalize_storage_type
from typing_extensions import Self

if TYPE_CHECKING:
    from litgpt import GPT
    import torch.distributed


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total: int = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)  # type: ignore
            else:
                total += p.numel()
    return total


def reset_parameters(module: nn.Module) -> None:
    """Calls `reset_parameters` on the module and all its submodules."""
    for mod in module.modules():
        if callable(getattr(mod, "reset_parameters", None)):
            mod.reset_parameters()


def check_valid_checkpoint_dir(checkpoint_dir: Path, model_filename: str = "lit_model.pth") -> None:
    files = {
        "lit_model.pth": (checkpoint_dir / "lit_model.pth").is_file(),
        "lit_config.json": (checkpoint_dir / "lit_config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file()
        or (checkpoint_dir / "tokenizer.model").is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at https://github.com/Lightning-AI/litgpt/blob/main/tutorials\n"
        f"{extra}\nSee all download options by running:\n litgpt download"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, reduce_args = tensor.__reduce_ex__(protocol_version)
        if reduce_args[0] == torch._utils._rebuild_tensor_v2:  # type: ignore # MAY NOT WORK FOR TORCH 2.5
            # for Tensors with Python attributes
            (a0, a1, (storage, *a2_other), *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (a0, a1, (storage_proxy, *a2_other), *other_reduce_args)
        else:
            (storage, *other_reduce_args) = reduce_args
            assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
            storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
            self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


T = TypeVar("T")


@torch.no_grad()
def batch_exact_match(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: list,
    cfg,  # litgpt.settings.CLISettings
    tokenizer,  # litgpt.tokenizer.Tokenizer
    step_i: int = None,
):
    """Function to compute exact match statistics for a batch of logits and labels.

    Args:
        logits: torch.Tensor of shape (batch_size, seq_len, vocab_size)
        labels: torch.Tensor of shape (batch_size, seq_len)
        metadata: list of metadata objects, one for each row in the batch
        cfg: CLISettings object containing the configuration for the run
        tokenizer: Tokenizer object containing the tokenizer used for the run

    Returns:
        em_stats: dict of dicts containing the exact match statistics for each prefix length and suffix length
        represented as a list of booleans, maximally containing the number of elements in the batch

    First, get the basic signal which checks the argmax of the logits against the labels
    creating a binary mask of correct/incorrect predictions (bsz, seq_len), where seq_len is the
    local, batchwise sequence length.
    Then, for each prefix len, for each suffix len, check whether the slice from the prefix_len to the
    prefix_len + suffix_len contains all 1s. If so, at test time, if the model had been
    given this prefix, it would have produced the correct output, because it would have
    predicted each token in the suffix correctly one by one.

    """
    em_stats = {pl: {sl: [] for sl in cfg.suffix_lengths} for pl in cfg.prefix_lengths}
    em_stats["natural_suffix"] = {sl: [] for sl in cfg.suffix_lengths}
    # we also add a special category for the unmasked regions of the labels
    # store with the suffixes, but with sl = "unmasked"
    em_stats["natural_suffix"]["unmasked"] = []

    for logits_row, labels_row, metadata_row in zip(logits, labels, metadata):

        correct_preds = torch.argmax(logits_row, dim=-1) == labels_row

        valid_mask = labels_row != tokenizer.pad_id

        only_tail_padding = torch.equal(
            labels_row[: labels_row[valid_mask].shape[0]], labels_row[valid_mask]
        )  # "Padding tokens should only be at the end of the row"

        # can proceed with prefix/suffix logic
        if only_tail_padding:
            stripped_preds = correct_preds[valid_mask]
            for prefix_len in cfg.prefix_lengths:
                for suffix_len in cfg.suffix_lengths:
                    if prefix_len + suffix_len > stripped_preds.shape[0]:
                        # then this would be a longer p+s than the row itself, this p+s is invalid for this row
                        continue
                    all_correct = torch.all(stripped_preds[prefix_len : prefix_len + suffix_len])
                    # em_stats[prefix_len][suffix_len].append(all_correct.item())
                    em_stats[prefix_len][suffix_len].append((all_correct.item(), metadata_row))
                    # if all_correct:
                    #     breakpoint()

            # we also want to consider the case where we measure just the last suffix_len tokens
            # with whatever the natural prefix would then be
            for suffix_len in cfg.suffix_lengths:
                if suffix_len > stripped_preds.shape[0]:
                    continue
                all_correct = torch.all(stripped_preds[-suffix_len:])
                em_stats["natural_suffix"][suffix_len].append((all_correct.item(), metadata_row))

        # We want to consider the case where we measure the unmasked tokens
        # which can be interleaved with the padding tokens in any way
        # in the case of no masking anywhere, this will report the full sequence all or nothing accuracy
        all_correct = torch.all(correct_preds[valid_mask])
        em_stats["natural_suffix"]["unmasked"].append((all_correct.item(), metadata_row))

    # if step_i is not None and step_i % 100 == 0:
    #     breakpoint()

    return em_stats


def flatten_dict(d, parentkey="", sep="_", coerce_to_str=False):
    """Not well tested, probably works fine for dicts of simple types and minimal depth."""
    items = []
    for k, v in d.items():
        if coerce_to_str:
            k = str(k)
        newkey = parentkey + sep + k if parentkey else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, newkey, sep=sep, coerce_to_str=coerce_to_str).items())
        else:
            items.append((newkey, v))
    return dict(items)


def reduce_memorization_metrics(
    em_stats_list: list[dict] = None,
    total_steps_taken: int = None,
    fabric: L.Fabric = None,
    cfg=None,  # litgpt.settings.CLISettings
):
    assert em_stats_list is not None
    assert total_steps_taken is not None
    assert fabric is not None
    assert cfg is not None

    total_em_stats = {pl: {sl: None for sl in cfg.suffix_lengths} for pl in cfg.prefix_lengths}
    total_em_stats["natural_suffix"] = {sl: None for sl in cfg.suffix_lengths}
    total_em_stats["natural_suffix"]["unmasked"] = None
    ds_wise_em_stats = {
        ds_metadata: {pl: {sl: None for sl in cfg.suffix_lengths} for pl in cfg.prefix_lengths}
        for ds_metadata in cfg.val_dataset_prefixes
    }
    for ds_metadata in cfg.val_dataset_prefixes:
        ds_wise_em_stats[ds_metadata]["natural_suffix"] = {sl: None for sl in cfg.suffix_lengths}
    ds_wise_em_stats[ds_metadata]["natural_suffix"]["unmasked"] = None

    ds_meta_to_int = {ds_metadata: i for i, ds_metadata in enumerate(cfg.val_dataset_prefixes)}
    ds_meta_from_int = {i: ds_metadata for i, ds_metadata in enumerate(cfg.val_dataset_prefixes)}
    for pl in cfg.prefix_lengths:
        for sl in cfg.suffix_lengths:
            for batch_em_stats in em_stats_list:
                # unzip the batch hits and metadata
                if len(batch_em_stats[pl][sl]) == 0:
                    # nothing to record here
                    continue
                batch_hits, batch_metadata = zip(*batch_em_stats[pl][sl])
                if total_em_stats[pl][sl] is None:
                    total_em_stats[pl][sl] = dict(hits=torch.tensor(batch_hits), metadata=batch_metadata)
                else:
                    total_em_stats[pl][sl] = dict(
                        hits=torch.cat([total_em_stats[pl][sl]["hits"], torch.tensor(batch_hits)]),
                        metadata=total_em_stats[pl][sl]["metadata"] + batch_metadata,
                    )
            if total_em_stats[pl][sl] is None:
                total_em_stats[pl][sl] = dict(hits=torch.tensor([]), metadata=[])

            # we need to pass tensors of the same size to all_gather, so we'll
            # convert the hits tensor to ints (0,1) and then pad with -1, and remove those after all_gather
            payload_tensor = torch.full(
                (total_steps_taken * cfg.micro_batch_size,), -1, dtype=torch.int32, device=fabric.device
            )
            payload_tensor[: total_em_stats[pl][sl]["hits"].shape[0]] = total_em_stats[pl][sl]["hits"].int()
            pl_sl_global_em = fabric.all_gather(payload_tensor).reshape(-1).tolist()
            pl_sl_global_em = [em for em in pl_sl_global_em if em != -1]

            total_em_stats[pl][sl]["metadata"] = torch.tensor(
                [ds_meta_to_int[meta] for meta in total_em_stats[pl][sl]["metadata"]]
            )
            meta_payload_tensor = torch.full(
                (total_steps_taken * cfg.micro_batch_size,), -1, dtype=torch.int32, device=fabric.device
            )
            meta_payload_tensor[: total_em_stats[pl][sl]["metadata"].shape[0]] = total_em_stats[pl][sl]["metadata"]

            pl_sl_global_metadata = fabric.all_gather(meta_payload_tensor).reshape(-1).tolist()
            pl_sl_global_metadata = [meta for meta in pl_sl_global_metadata if meta != -1]

            pl_sl_global_metadata = [ds_meta_from_int[meta] for meta in pl_sl_global_metadata]

            total_em_stats[pl][sl] = dict(
                em_rate=torch.tensor(pl_sl_global_em).float().mean().item(),
                samples=len(pl_sl_global_em),
            )
            # now transpose this to the dataset-wise stats by groupbying hits on metadata
            for ds_metadata in cfg.val_dataset_prefixes:
                ds_hits = [hit for hit, meta in zip(pl_sl_global_em, pl_sl_global_metadata) if meta == ds_metadata]
                if len(ds_hits) == 0:
                    continue
                ds_wise_em_stats[ds_metadata][pl][sl] = dict(
                    em_rate=torch.tensor(ds_hits).float().mean().item(),
                    samples=len(ds_hits),
                )
    # now handle the natural suffix case
    # for sl in cfg.suffix_lengths:
    for sl in cfg.suffix_lengths + ["unmasked"]:
        for batch_em_stats in em_stats_list:
            if len(batch_em_stats["natural_suffix"][sl]) == 0:
                continue
            batch_hits, batch_metadata = zip(*batch_em_stats["natural_suffix"][sl])
            if total_em_stats["natural_suffix"][sl] is None:
                total_em_stats["natural_suffix"][sl] = dict(hits=torch.tensor(batch_hits), metadata=batch_metadata)
            else:
                total_em_stats["natural_suffix"][sl] = dict(
                    hits=torch.cat([total_em_stats["natural_suffix"][sl]["hits"], torch.tensor(batch_hits)]),
                    metadata=total_em_stats["natural_suffix"][sl]["metadata"] + batch_metadata,
                )
        if total_em_stats["natural_suffix"][sl] is None:
            total_em_stats["natural_suffix"][sl] = dict(hits=torch.tensor([]), metadata=[])

        # we need to pass tensors of the same size to all_gather, so we'll
        # convert the hits tensor to ints (0,1) and then pad with -1, and remove those after all_gather
        payload_tensor = torch.full(
            (total_steps_taken * cfg.micro_batch_size,), -1, dtype=torch.int32, device=fabric.device
        )
        payload_tensor[: total_em_stats["natural_suffix"][sl]["hits"].shape[0]] = total_em_stats["natural_suffix"][sl][
            "hits"
        ].int()

        sl_global_em = fabric.all_gather(payload_tensor).reshape(-1).tolist()
        sl_global_em = [em for em in sl_global_em if em != -1]

        total_em_stats["natural_suffix"][sl]["metadata"] = torch.tensor(
            [ds_meta_to_int[meta] for meta in total_em_stats["natural_suffix"][sl]["metadata"]]
        )
        meta_payload_tensor = torch.full(
            (total_steps_taken * cfg.micro_batch_size,), -1, dtype=torch.int32, device=fabric.device
        )
        meta_payload_tensor[: total_em_stats["natural_suffix"][sl]["metadata"].shape[0]] = total_em_stats[
            "natural_suffix"
        ][sl]["metadata"]

        sl_global_metadata = fabric.all_gather(meta_payload_tensor).reshape(-1).tolist()
        sl_global_metadata = [meta for meta in sl_global_metadata if meta != -1]

        sl_global_metadata = [ds_meta_from_int[meta] for meta in sl_global_metadata]

        total_em_stats["natural_suffix"][sl] = dict(
            em_rate=torch.tensor(sl_global_em).float().mean().item(),
            samples=len(sl_global_em),
        )
        for ds_metadata in cfg.val_dataset_prefixes:
            ds_hits = [hit for hit, meta in zip(sl_global_em, sl_global_metadata) if meta == ds_metadata]
            if len(ds_hits) == 0:
                continue
            ds_wise_em_stats[ds_metadata]["natural_suffix"][sl] = dict(
                em_rate=torch.tensor(ds_hits).float().mean().item(),
                samples=len(ds_hits),
            )

    return total_em_stats, ds_wise_em_stats


global hash_table
hash_table = None
table_size = 1_000_003


def _load_hash_table(device):
    global hash_table
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)  # fib47 is prime
    hash_table = torch.rand(table_size, device=device, generator=rng)


def apply_gl(
    targets: torch.Tensor,
    strategy: str,
    k: int,
    gl_start_position: int,
    ignore_index,
    gl_context_width: int = 4,  # context width for hash based drops
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a mask to a tensor to ignore every k-th token.
    `targets` is NOT updated in-place so apply_gl can be indepdently called for analysis/debugging/logging.
    Args:
        target: The target to apply the GL mask to.
        strategy: The strategy to use for GL.
            options implemented:
                - "static": Ignore every k-th token starting from `gl_start_position`.
                - "hash-legacy": Ignore tokens based on a hash of the context. For debugging purposes only.
                - "hash-table": Ignore tokens based on a hash of the context using a precomputed table.
                - "hash-avalanche": Ignore tokens based on a hash of the context using a hash function.
        k: The frequency with which tokens are ignored?
        gl_start_position: The position to start ignoring tokens from.
        context_width: Context width for hash-based approaches.
    Returns:
        The target with the mask applied and the indices of the dropped tokens.
    """
    device = targets.device
    mbs, block_size = targets.shape
    masked_targets = targets.clone()

    if strategy == "static":
        dropped_token_indices = torch.arange(block_size, device=device)[gl_start_position::k].long()
        masked_targets[:, dropped_token_indices] = ignore_index
    elif strategy == "seeded_random":
        random_tensor = torch.randint(1, k + 1, size=targets.size())
        dropped_token_indices = (random_tensor == k).int()  # probability of dropping a token is 1/k
        masked_targets[dropped_token_indices] = ignore_index
    elif strategy == "random_but_consisent_for_given_micro_batch":  # long name but better than unclear
        # TODO confirm that for given sample, its microbatch neighbours remains same through epochs
        micro_batch_seed = targets.sum().item()
        generator = torch.Generator().manual_seed(micro_batch_seed)
        random_tensor = torch.randint(1, k + 1, size=targets.size(), generator=generator)
        dropped_token_indices = (random_tensor == k).int()  # probability of dropping a token is 1/k
        masked_targets[dropped_token_indices] = ignore_index
    elif strategy == "hash-legacy":
        # Old hash for sanity checks, do not use
        dropped_token_indices = torch.zeros_like(targets)
        rng = torch.Generator(device=device)
        for b in range(mbs):
            for s in range(gl_context_width, block_size):
                prf_key = targets[b, s - gl_context_width : s].prod()
                rng.manual_seed(prf_key.item() % (2**64 - 1))
                dropped_token_indices[b, s] = torch.rand((1,), device=device) < 1 / k
        masked_targets[dropped_token_indices] = ignore_index
    elif strategy == "hash-table":
        global hash_table
        if hash_table is None:
            _load_hash_table(device)
        hashed_keys = hash_table[targets.unfold(1, gl_context_width, 1).prod(dim=-1) % table_size]
        dropped_token_indices = hashed_keys < 1 / k
        masked_targets[:, gl_context_width - 1 :][dropped_token_indices] = ignore_index
        dropped_token_indices = dropped_token_indices.int()
    elif strategy == "hash-avalanche":
        keys = targets.unfold(1, gl_context_width, 1).prod(dim=-1).to(dtype=torch.uint64)
        hashed_keys = hashint(keys, width=32).long()
        dropped_token_indices = hashed_keys < ((1 << 32) - 1) / k
        masked_targets[:, gl_context_width - 1 :][dropped_token_indices] = ignore_index
    else:
        raise NotImplementedError(f"{strategy} GL strategy is not implemented. Try 'static' instead.")

    return masked_targets, dropped_token_indices


@torch.compile  # required for uint64 support
def hashint(key: torch.Tensor, width: int = 32):
    """
    For any 1<k<=64, let mask=(1<<k)-1. hash_64() is a bijection on [0,1<<k), which means
    hash_64(x, mask)==hash_64(y, mask) if and only if x==y. hash_64i() is the inversion of
    hash_64(): hash_64i(hash_64(x, mask), mask) == hash_64(hash_64i(x, mask), mask) == x.
    """
    # thomas wang 64bit
    mask = (1 << width) - 1
    key = (~key + (key << 21)) & mask
    key = (key << 21) - key - 1
    key = key ^ key >> 24
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key * 265
    key = key ^ key >> 14
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key * 21
    key = key ^ key >> 28
    key = (key + (key << 31)) & mask
    return key


def slice_logits_remap_labels(
    logits: torch.Tensor, targets: torch.Tensor, target_range: tuple, ignore_indices: list[int] = [-1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slices logits to a specific index range, and remaps target labels to the new range, zero indexed."""
    # provided range is _inclusive_
    # first we slice the logits to just that range (+1)
    # considering all elms in the targets that are not ignored
    # we remap them to the new range. eg if the range is 32003,32005
    # we slice the logits to logits[:, 32003:32006]
    # and we remap 32003 to 0, 32004 to 1, 32005 to 2

    # assert len(target_range) == 2, "target_range must be a tuple of two integers"
    # assert target_range[0] < target_range[1], "target_range must be a tuple of two unique integers in ascending order"
    # assert target_range[0] >= 0, "target_range[0] must be a non-negative integer"
    # assert target_range[1] < logits.size(-1), "target_range[1] must be less than the number of logits"

    assert min(target_range) >= 0, "smallest target_range value must be non-negative"
    assert max(target_range) < logits.size(-1), "largest target_range value must be less than the number of logits"
    assert sorted(target_range) == list(target_range), "target_range must be sorted already, sanity convention"

    label_id_to_tgt_id = {label_id: tgt_id for tgt_id, label_id in enumerate(target_range)}

    # # slice logits
    # start, end = target_range
    # logits = logits[:, start : end + 1]
    # # remap targets
    # for tgt_id in range(start, end + 1):
    #     targets[targets == tgt_id] = tgt_id - start

    # # create a mask to select only the specified elems from the logits tensor
    # mask = torch.zeros(logits.size(-1), dtype=torch.bool, device=logits.device)
    # mask[target_range] = True

    # logits = logits[:, mask]

    # oof, mask might have broken gradient flow?
    # we can still use slice/select mechanics to get the desired logits
    # though this is unlikely to have caused any difference.
    logits = logits[:, target_range]

    # remap the targets using the mapping, rather than offsets because they need not be contiguous
    for label_id, tgt_id in label_id_to_tgt_id.items():
        targets[targets == label_id] = tgt_id

    return logits, targets


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_indices: list[Optional[int]] = [None],
    reduction: Optional[str] = None,
    training: bool = True,
    z_loss_eps=0.0,
    gl_k: int = None,
    gl_strategy: Union[str, None] = None,
    gl_start_position: int = None,
    gl_context_width: int = None,
    target_range: tuple = None,
    return_logits_targets: bool = None,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude
    ignore_index = -100 if (ignore_indices[0] is None) else ignore_indices[0]  # fixed, credit ywen :<
    for additional_ignore in ignore_indices[1:]:
        if additional_ignore is not None and additional_ignore != ignore_index:
            targets[targets == additional_ignore] = ignore_index

    # ignore every k-th token using ignore_index for Token Loss Dropout (Goldfish Loss/GL)
    if gl_strategy is not None and training:
        targets, _ = apply_gl(targets, gl_strategy, gl_k, gl_start_position, ignore_index, gl_context_width)

    cross_entropy_fn = partial(
        torch.nn.functional.cross_entropy,
        ignore_index=ignore_index,
    )

    bs = targets.shape[0]
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    if target_range is not None:
        # print("Using slice loss!")
        logits, targets = slice_logits_remap_labels(logits, targets, target_range, ignore_indices)
    else:
        # print("Using normal loss!")
        pass

    if chunk_size == 0:
        # no chunking at all
        if reduction is not None:
            loss = cross_entropy_fn(input=logits, target=targets, reduction=reduction)
        else:
            loss = cross_entropy_fn(input=logits, target=targets)
    else:
        # chunk cross entropy
        logit_chunks = logits.split(chunk_size)
        target_chunks = targets.split(chunk_size)
        losses = torch.zeros_like(targets, dtype=logits.dtype, device=logits.device)  # prealloc required for compile

        for idx, (logit_chunk, target_chunk) in enumerate(zip(logit_chunks, target_chunks)):
            loss_chunk = cross_entropy_fn(input=logit_chunk, target=target_chunk, reduction="none")
            losses[idx * chunk_size : (idx + 1) * chunk_size] = loss_chunk

        non_masked_elems = (targets != ignore_index).sum().clamp(min=1.0)
        if reduction is not None:
            loss = losses.sum() / non_masked_elems
        else:
            # seq wise loss
            losses = losses.reshape(bs, -1)
            valid_mask = (targets != ignore_index).float()
            valid_mask = valid_mask.reshape(bs, -1)
            loss = (losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)

    if z_loss_eps > 0.0 and training:
        loss += z_loss_eps * torch.logsumexp(logits, dim=-1).pow(2).mean()

    if not return_logits_targets:
        return loss
    else:
        non_masked_mask = targets != ignore_index
        return loss, logits[non_masked_mask, :], targets[non_masked_mask]


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def load_checkpoint(fabric: L.Fabric, model: nn.Module, checkpoint_path: str, strict: bool = True) -> None:
    if isinstance(fabric.strategy, FSDPStrategy):
        fabric.load_raw(checkpoint_path, model, strict=strict)
    else:
        state_dict = lazy_load(checkpoint_path)
        state_dict = state_dict.get("model", state_dict)
        model.load_state_dict(state_dict, strict=strict)


def flops_per_param(max_seq_length: int, n_layer: int, n_embd: int, n_params: int) -> int:
    flops_per_token = 2 * n_params  # each parameter is used for a MAC (2 FLOPS) per network operation
    # this assumes that all samples have a fixed length equal to the block size
    # which is most likely false during finetuning
    flops_per_seq = flops_per_token * max_seq_length
    attn_flops_per_seq = n_layer * 2 * 2 * (n_embd * (max_seq_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_flops(model: "GPT", training: bool) -> int:
    """Measures estimated FLOPs for MFU.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    # using all parameters for this is a naive over estimation because not all model parameters actually contribute to
    # this FLOP computation (e.g. embedding, norm). For this reason, the result will be higher by a fixed percentage
    # (~10%) compared to the measured FLOPs, making those lower but more realistic.
    # For a proper estimate, this needs a more fine-grained calculation as in Appendix A of the paper.
    n_trainable_params = num_parameters(model, requires_grad=True)
    trainable_flops = flops_per_param(
        model.max_seq_length, model.config.n_layer, model.config.n_embd, n_trainable_params
    )
    # forward + backward + gradients (assumes no gradient accumulation)
    ops_per_step = 3 if training else 1
    n_frozen_params = num_parameters(model, requires_grad=False)
    frozen_flops = flops_per_param(model.max_seq_length, model.config.n_layer, model.config.n_embd, n_frozen_params)
    # forward + backward
    frozen_ops_per_step = 2 if training else 1
    return ops_per_step * trainable_flops + frozen_ops_per_step * frozen_flops


def simple_gptneox_tflops(metrics, fabric, cfg, batch_size=None, iter_time_s=None):
    """Estimate the TFLOPs using the GPT-NeoX napkin math.

    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/logging.py#L82
    Think about significant approximations and potential correctness issues under FSDP or AxoNN.
    """

    world_size = fabric.world_size
    vocab_size = cfg.model_config.vocab_size
    batch_size = cfg.world_batch_size if batch_size is None else batch_size
    seq_len = cfg.block_size
    hidden_size = cfg.model_config.n_embd
    num_layers = cfg.model_config.n_layer
    ckpt_activations_factor = 4 if cfg.gradient_checkpointing else 3
    flops_per_iteration = (
        24
        * ckpt_activations_factor
        * batch_size
        * seq_len
        * num_layers
        * (hidden_size**2)
        * (1.0 + (seq_len / (6.0 * hidden_size)) + (vocab_size / (16.0 * num_layers * hidden_size)))
    )
    return (flops_per_iteration / (iter_time_s * world_size)) / 1e12


def simple_axonn_tflops(metrics, fabric, cfg, batch_size=None, iter_time_s=None):
    """This function copied in from megatron_logging.py.
    The GemmaMLP catch is probably wrong. Unsure about usage."""

    world_size = fabric.world_size
    config = cfg.model_config

    N = config.n_layer
    B = batch_size
    S = config.block_size
    V = config.padded_vocab_size
    H = config.n_embd
    IH = config.intermediate_size

    if config.mlp_class_name == "LLaMAMLP" or config.mlp_class_name == "GemmaMLP":
        linear_flops = N * (32 * B * S * H * H + 24 * B * S * H * IH)
    elif config.mlp_class_name == "GptNeoxMLP":
        linear_flops = N * (32 * B * S * H * H + 16 * B * S * H * IH)
    else:
        raise NotImplementedError
    attention_flops = N * (16 * B * S * S * H)
    head_flops = 6 * B * S * H * V
    if cfg.gradient_checkpointing:
        flops = linear_flops + attention_flops + head_flops
    else:
        flops = 3 / 4 * (linear_flops + attention_flops) + head_flops

    return flops / 1e12 / iter_time_s / world_size


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def copy_config_files(source_dir: Path, out_dir: Path) -> None:
    """Copies the specified configuration and tokenizer files into the output directory."""

    config_files = ["generation_config.json", "model_config.yaml"]
    tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

    for file_name in config_files + tokenizer_files:
        src_path = source_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, out_dir)


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_config_read_mode, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    kwargs.setdefault("as_positional", False)

    return CLI(*args, **kwargs)


def capture_hparams() -> Dict[str, Any]:
    """Captures the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back  # type: ignore
    locals_of_caller = caller_frame.f_locals  # type: ignore
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)  # type: ignore
        else:
            hparams[name] = str(value)
    return hparams


def save_hyperparameters(function: Callable, checkpoint_dir: Path) -> None:
    """Captures the CLI parameters passed to `function` without running `function` and saves them to the checkpoint."""
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    known_commands = [
        ("finetune", "full"),
        ("finetune", "lora"),
        ("finetune", "adapter"),
        ("finetune", "adapter_v2"),
        ("pretrain",),
    ]
    for known_command in known_commands:
        unwanted = slice(1, 1 + len(known_command))
        if tuple(sys.argv[unwanted]) == known_command:
            sys.argv[unwanted] = []

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / "hyperparameters.yaml", overwrite=True)


def save_config(config, checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project=name, resume=resume, **kwargs)
    else:
        raise ValueError(
            f"`--logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'."
        )


def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (
        not checkpoint_dir.is_dir()
        and checkpoint_dir.parts[0] != "checkpoints"
        and not checkpoint_dir.is_absolute()
        and new_checkpoint_dir.exists()
    )
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir


def fsdp_auto_wrap_policy(set_of_transformer_layers: set[Type[torch.nn.Module]]):
    import functools
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=set_of_transformer_layers
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])

    return auto_wrap_policy


# wrapper for lightning fabrics
class LightningFabric:
    def __init__(self, devices, strategy, precision, loggers=[], num_nodes: int = 1):

        fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=loggers, num_nodes=num_nodes)
        self._underlying_fabric = fabric
        self.strategy = strategy
        self._precision = precision
        self.strategy_name = strategy.__class__.__name__
        self._note_clip_strategy_once = False

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._underlying_fabric, attr)

    def get_prefix_for_checkpoint(self):
        return f"checkpoints-{self.strategy_name}"

    def log_to_summary(self, dict_of_values):
        if self.global_rank == 0:
            for key, value in dict_of_values.items():
                self.loggers[0].experiment.summary[key] = value

    def clip_gradients(self, model, optimizer, max_norm):
        if "AxonnStrategy" in self.strategy_name:
            if self._note_clip_strategy_once is False:
                print("Using clip_gradients_norm from AxonnStrategy.")
                self._note_clip_strategy_once = True
            # call the implementation directly from the AxonnStrategy
            grad_norm = self.strategy.clip_gradients_norm(
                model, optimizer, max_norm=max_norm
            )  # , norm_type=2.0, error_if_nonfinite=False
        else:
            grad_norm = self._underlying_fabric.clip_gradients(
                model, optimizer, max_norm=max_norm
            )  # clip_val=None, norm_type=2.0, error_if_nonfinite=True
        return grad_norm

    def setup(self, model: torch.nn.Module, compile: bool = False, compile_ddp: bool = True):
        """Compiling DDP turns out not to improve speed on the A6000 ada cards"""

        if "DDP" in self.strategy_name and compile and compile_ddp:
            # the default dtype needs to be set before the lightning fabric handles it, otherwise DDP cannot be compiled
            # due to default dtypes changes in the compiled program
            if "bf16-true" in self._precision:
                torch.set_default_dtype(torch.bfloat16)
            elif "16-true" in self._precision:
                torch.set_default_dtype(torch.float16)

            model = self._underlying_fabric.setup(model)
            if compile:
                model = torch.compile(model)  # type: ignore
        else:
            if compile:
                # error on dynamic shape
                model = torch.compile(model)  # type: ignore
            model = self._underlying_fabric.setup(model)

        return model


class SimpleFabric:
    """Simple ddp-based fabric without lightning cruft, can be a template for other fabrics"""

    def __init__(
        self,
        precision="bf16-true",
        loggers=None,
        local_device_init=True,
    ):
        self.precision = precision
        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.local_device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            device_id=self.local_device if local_device_init else None,  # this immediately forms the NCCL communicator
        )
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]

        if precision == "bf16-true":
            self.dtype = torch.bfloat16
        elif precision in ["fp16-mixed", "bf16-mixed"]:
            self.dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision type {precision} provided.")
        # Please be careful with vague constructors with this turned on, float32 constructors should say so:
        torch.set_default_dtype(self.dtype)
        torch.cuda.set_device(self.local_device)
        self.global_rank_for_creating_dataloader = self.rank
        self.world_size = torch.distributed.get_world_size()

    def launch(self):
        pass

    def print(self, msg, ranks=[0]):
        if torch.distributed.get_rank() in ranks:
            print(msg, flush=True)

    def log_dict(self, metrics, step=None):
        if self.rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self._loggers:
                logger.log_metrics(metrics=metrics, step=step)

    def log_to_summary(self, dict_of_values):
        if self.rank == 0:
            for key, value in dict_of_values.items():
                self.logger.experiment.summary[key] = value

    @property
    def device(self):
        return self.local_device

    @property
    def global_rank(self):
        return self.rank

    @property
    def logger(self):
        return self._loggers[0]

    @property
    def strategy(self):
        return "SimpleDDP"

    def setup_dataloaders(self, train_dataloader, val_dataloader):
        return train_dataloader, val_dataloader

    def setup(self, model, compile: bool = False):
        model = model.to(self.device)
        # wrap DDP and AMP
        self._ddp_model_ref = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        self.model_ref = _FabricModule(
            forward_module=self._ddp_model_ref,
            original_module=model,
            precision=self.precision,
        )
        # compile after DDP to compile DDP calls
        if compile:
            self.model_ref = torch.compile(self.model_ref)
        return self.model_ref

    def setup_optimizers(self, optimizers):
        return optimizers

    def all_reduce(self, data, reduce_op="mean") -> torch.Tensor:
        """
        All reduce over data parallel groups
        """
        op = torch.distributed.ReduceOp.AVG
        if reduce_op == "sum":
            op = torch.distributed.ReduceOp.SUM
        data = convert_to_tensors(data, device=self.device)

        torch.distributed.all_reduce(data, op=op)
        return data

    def barrier(self):
        torch.distributed.barrier(device_ids=[self.device.index])

    def broadcast(self, obj, src: int = 0):
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]

    def backward(self, loss, model=None):
        loss.backward()

    def clip_gradients(self, model, optimizer, max_norm):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0)
        return grad_norm

    def save(self, checkpoint_file, state, overwrite=True):
        if self.rank == 0:
            save_state_dict = {}
            for key, value in state.items():
                if key not in ["optimizer", "model"]:
                    save_state_dict[key] = value
                else:
                    save_state_dict[key] = value.state_dict()

            checkpoint_folder = os.path.dirname(checkpoint_file)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            if os.path.exists(checkpoint_file) and not overwrite:
                raise ValueError(f"Checkpoint {checkpoint_file} already exists")
            torch.save(save_state_dict, checkpoint_file)

    def load(self, resume, state):
        checkpoint = torch.load(resume)
        for key, value in checkpoint.items():
            if key not in ["optimizer", "model"]:
                state[key] = value
            else:
                if key == "model":
                    state[key].load_state_dict(value)
                else:
                    state[key].load_state_dict(value)

    @contextmanager
    def no_backward_sync(self, model, enabled=True):
        with self._ddp_model_ref.no_sync() if enabled else nullcontext():
            yield

    def get_prefix_for_checkpoint(self):
        return "checkpoints-sane-ddp"

    @contextmanager
    def init_module(self, empty_init=False):
        yield None

    def seed_everything(self, seed: Optional[int] = None, workers: bool = False):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if seed is None:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            if env_seed is None:
                seed = 0
                self.print(f"Warning: No seed found, seed set to {seed}")
            else:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = 0
                    self.print(f"Warning: Invalid seed found: {repr(env_seed)}, seed set to {seed}")
        elif not isinstance(seed, int):
            seed = int(seed)

        if not (min_seed_value <= seed <= max_seed_value):
            self.print(f"Warning: {seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
            seed = 0

        # print(rank_prefixed_message(f"Seed set to {seed}", self.global_rank))
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

        return seed


class _FabricModule(torch.nn.Module):
    def __init__(self, forward_module, precision, original_module=None) -> None:
        """The FabricModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            forward_module: The module to wrap the ``forward`` method on.
            strategy: Reference to the strategy for handling precision etc.
            original_module: The original, unmodified module as passed into the
                :meth:`lightning.fabric.fabric.Fabric.setup` method. This is needed when attribute lookup
                on this wrapper should pass through to the original module.

        """
        super().__init__()
        self._forward_module = forward_module
        self._original_module = original_module or forward_module
        self.precision = precision

        class AmpSettings(TypedDict):
            device_type: str
            dtype: torch.dtype
            enabled: bool

        self.amp_settings: AmpSettings = {
            "device_type": "cuda",
            "enabled": "mixed" in precision,
            "dtype": torch.bfloat16 if "bf16" in precision else torch.float16,
        }

    @property
    def module(self):
        return self._original_module or self._forward_module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward method."""
        with torch.autocast(**self.amp_settings):
            output = self._forward_module(*args, **kwargs)
        return output

    def __getattr__(self, item: Any) -> Any:
        try:
            # __getattr__ gets called as a last resort if the attribute does not exist
            # call nn.Module's implementation first
            return super().__getattr__(item)
        except AttributeError:
            # If the attribute is not available on the _FabricModule wrapper, redirect to the wrapped nn.Module
            original_module = super().__getattr__("_original_module")
            attr = getattr(original_module, item)
            return attr


def param_count_estimator(
    width=None, depth=None, vocab_size=None, n_head=None, head_size=None, n_query_groups=None, intermediate_size=None
):
    # Embedding layer parameters
    embedding_params = vocab_size * width

    # Attention parameters: attn + proj
    attn_shape = (n_head + 2 * n_query_groups) * head_size
    attn_params = (width * attn_shape) + (head_size * n_head * width)

    # MLP parameters: fc_1 + fc_2 + proj
    mlp_params = (width * intermediate_size) + (width * intermediate_size) + (intermediate_size * width)

    # RMSNorm parameters: 2 per block + 1 final norm
    norm_params_per_block = 2 * width

    # Total per block
    total_block_params = attn_params + mlp_params + norm_params_per_block

    # All layers (blocks)
    total_params = total_block_params * depth

    # Final LayerNorm and LM Head
    final_norm_params = width
    lm_head_params = width * vocab_size

    # Total model parameters
    return total_params + embedding_params + final_norm_params + lm_head_params
