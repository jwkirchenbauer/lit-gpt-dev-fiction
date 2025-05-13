import gc
import torch
import argparse
import os
from tqdm import tqdm
import json

from litgpt.utils import incremental_save, lazy_load

# BEGIN HECK
# as a hack we need to be able to get utils from the main training script
# so we add the repo root to the python path
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from train import *

# END HECK


def get_latest_checkpoint(folder_path):
    checkpoints = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1].split("-")[0]))
    return latest_checkpoint


def get_matching_checkpoint(folder_path, keystring):
    checkpoints = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not checkpoints:
        return None

    matching_checkpoint = None
    for checkpoint in checkpoints:
        if keystring in checkpoint:
            matching_checkpoint = checkpoint
            break

    return matching_checkpoint


def is_lm_head_weight(key):
    return "lm_head.weight" in key


def is_wte_weight(key):
    return "wte.weight" in key


def is_tensor_parallel_weight(key):
    if is_lm_head_weight(key):
        return True

    if is_wte_weight(key):
        return True

    if "transformer.h" in key:
        if "weight" in key and "norm" not in key:
            return True
    return False


def get_shape(key, n_embd):
    if "attn.attn.weight" in key:
        # return n_embd, -1
        return -1, n_embd
    elif "attn.proj.weight" in key:
        return n_embd, -1
    elif ("mlp.fc_1.weight" in key) or ("mlp.fc_2.weight" in key) or ("mlp.fc.weight" in key):
        # return n_embd, -1
        return -1, n_embd
    elif "mlp.proj.weight" in key:
        # return -1, n_embd
        return n_embd, -1
    elif "lm_head.weight" in key:
        # return n_embd, -1
        return -1, n_embd
    elif "wte.weight" in key:
        # return n_embd, -1
        return -1, n_embd
    else:
        raise ValueError("This is some weird tensor parallel weight")


def load_sharded_and_save_unified(combined_checkpoint, args, run_dir=None, keystring=None):

    with open(os.path.join(args.out_dir, "run_config.json")) as f:
        run_config = json.load(f)

    with open(os.path.join(args.out_dir, "model_config.json")) as f:
        model_config = json.load(f)

    n_embd = model_config["n_embd"]
    tensor_parallel_size = run_config["fabric"]["depth_tensor_parallel_size"]
    print("Loading checkpoint shards ...")
    checkpoints = []
    latest_checkpoint_filename = None
    for tp_shard in tqdm(range(tensor_parallel_size)):
        path = os.path.join(run_dir, f"checkpoints-AxonnStrategy/tp_row_0_col_0_depth_{tp_shard}/")
        if not os.path.exists(path):
            raise ValueError(f"Tensor Parallel Shard number {tp_shard} not found in {run_dir}")

        if keystring is not None:
            checkpoint_file_name = get_matching_checkpoint(path, keystring)
        else:
            checkpoint_file_name = get_latest_checkpoint(path)

        if latest_checkpoint_filename is None:
            latest_checkpoint_filename = checkpoint_file_name
        else:
            assert latest_checkpoint_filename == checkpoint_file_name, "some discrepancy in model checkpoints"

        if tp_shard == 0:
            print(f"Exporting checkpoint from tp_shard 0 from {checkpoint_file_name}")

        checkpoints.append(lazy_load(os.path.join(path, latest_checkpoint_filename)))

    keys = list(checkpoints[0]["model"].keys())

    for key in keys:
        if is_tensor_parallel_weight(key):
            to_concat = [checkpoint["model"][key]._load_tensor() for checkpoint in checkpoints]
            combined_checkpoint["model"][key] = torch.cat(to_concat).view(get_shape(key, n_embd))
            print(key, combined_checkpoint["model"][key].shape)
        else:
            combined_checkpoint["model"][key] = checkpoints[0]["model"][key]._load_tensor()

        for checkpoint in checkpoints:
            del checkpoint["model"][key]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output path for your experiment"
        " - it is assumed that model_config.json, run_config.json, and the tensor parallel checkpoints",
    )
    parser.add_argument(
        "--combined_ckpt_subdir",
        required=False,
        help="Subdirectory to save the combined checkpoint",
        default=None,
    )
    parser.add_argument(
        "--combined_ckpt_name",
        required=False,
        help="Name of the combined checkpoint file",
        default="combined_checkpoint.pth",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the combined checkpoint if it already exists",
    )
    parser.add_argument(
        "--keystring",
        required=False,
        default=None,
        help="Keystring to filter the checkpoints by",
    )

    args = parser.parse_args()

    # initialize a new empty state dict to hold our new weights
    combined_checkpoint = {"model": {}}

    input_run_dir = args.out_dir
    checkpoint_path = args.out_dir

    if args.combined_ckpt_subdir is not None:
        checkpoint_path = os.path.join(checkpoint_path, args.combined_ckpt_subdir)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    combined_checkpoint_file = os.path.join(checkpoint_path, args.combined_ckpt_name)
    if not combined_checkpoint_file.endswith(".pth"):
        combined_checkpoint_file += ".pth"

    if not args.overwrite:
        assert not os.path.exists(
            combined_checkpoint_file
        ), f"A combined checkpoint already exists at {combined_checkpoint_file}"
    else:
        # rm the existing file if it exists
        if os.path.exists(combined_checkpoint_file):
            os.remove(combined_checkpoint_file)

    with incremental_save(combined_checkpoint_file) as saver:
        load_sharded_and_save_unified(combined_checkpoint, args, run_dir=input_run_dir, keystring=args.keystring)
        gc.collect()
        saver.save(combined_checkpoint)

    print(f"Combined checkpoint successfully saved at {combined_checkpoint_file}!")
