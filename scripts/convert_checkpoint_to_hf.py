import json
import shutil
import sys
from pathlib import Path

import torch

from convert_pretrained_checkpoint import convert_checkpoint
from convert_lit_checkpoint import convert_lit_checkpoint

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import create_repo

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# BEGIN HECK
# as a hack we need to be able to get utils from the main training script
# so we add the repo root to the python path
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

# this captures things like the scale_lr function and stuff
from train import *

# END HECK

from litgpt.utils import CLI


@torch.inference_mode()
def convert_checkpoint_to_hf(
    checkpoint_file: Path = None,
    tokenizer_dir: Path = None,
    model_name: str = None,
    parent_dir: Path = None,
    axonn_patch: bool = False,
    q2_codebase: bool = True,
    push_to_hub: bool = True,
    hf_token: str = None,
    resize_embeddings: bool = False,  # may be required if vocab/embeds padded out to a multiple.
    from_lit_checkpoint: bool = False,  # if True, will convert from a lit checkpoint, otherwise from a training checkpoint
    convert_to_hf: bool = True,  # if False, will only convert to lit checkpoint
    embed_lit_config: bool = False,  # if True, will embed the lit config in the hf config
    hf_config_src: str = "hub",  # "tokenizer", "hub", "path"
    hf_config_path: Path = None,  # path to hf config file, if hf_config_src is "path"
) -> None:

    assert (
        checkpoint_file is not None or parent_dir is not None
    ), "Either checkpoint_file or parent_dir must be provided."
    assert model_name is not None, "model_name must be provided."

    parent_dir = checkpoint_file.parent.absolute() if parent_dir is None else parent_dir

    # if tokenizer_dir not provided, look it up in the run_config.json file
    if tokenizer_dir is None:
        assert parent_dir is not None, "parent_dir must be provided if tokenizer_dir is not provided"
        with open(parent_dir / "run_config.json") as f:
            run_config = json.load(f)
            tokenizer_dir = Path(run_config["tokenizer_path"])
            print(f"using inferred tokenizer_dir from run_config.json: {tokenizer_dir}")

    if not from_lit_checkpoint:
        ### convert training checkpoint to lit checkpoint
        with open(parent_dir / "model_config.json") as f:
            model_config = json.load(f)
        config_name = model_config["name"]
        convert_checkpoint(checkpoint_file, tokenizer_dir, config_name, parent_dir / f"lit_checkpoint_{model_name}")
    else:
        model_config = json.load(open(parent_dir / f"lit_checkpoint_{model_name}/lit_config.json"))
        # reindent this file
        with open(parent_dir / f"lit_checkpoint_{model_name}/lit_config.json", "w") as f:
            json.dump(model_config, f, indent=4)

    if not convert_to_hf:
        return
    ### convert lit checkpoint to hf checkpoint
    convert_lit_checkpoint(
        parent_dir / f"lit_checkpoint_{model_name}/lit_model.pth",
        parent_dir / f"hf_checkpoint_{model_name}/pytorch_model.bin",
        parent_dir / f"lit_checkpoint_{model_name}/lit_config.json",
        axonn_patch=axonn_patch,
        q2_codebase=q2_codebase,
        resize_embeddings=resize_embeddings,
    )

    for tokenizer_file in tokenizer_dir.glob("tokenizer*"):
        shutil.copyfile(tokenizer_file, parent_dir / f"hf_checkpoint_{model_name}" / tokenizer_file.name)

    if (tokenizer_dir / "generation_config.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "generation_config.json",
            parent_dir / f"hf_checkpoint_{model_name}" / "generation_config.json",
        )

    if (tokenizer_dir / "special_tokens_map.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "special_tokens_map.json",
            parent_dir / f"hf_checkpoint_{model_name}" / "special_tokens_map.json",
        )

    if (tokenizer_dir / "added_tokens.json").is_file():
        shutil.copyfile(
            tokenizer_dir / "added_tokens.json", parent_dir / f"hf_checkpoint_{model_name}" / "added_tokens.json"
        )

    if hf_config_src == "tokenizer":
        if (tokenizer_dir / "config.json").is_file():
            shutil.copyfile(tokenizer_dir / "config.json", parent_dir / f"hf_checkpoint_{model_name}" / "config.json")
    elif hf_config_src == "hub":
        hf_org = model_config["hf_config"]["org"]
        hf_name = model_config["hf_config"]["name"]
        hf_config = AutoConfig.from_pretrained(f"{hf_org}/{hf_name}")
        hf_config = hf_config.to_dict()
        with open(parent_dir / f"hf_checkpoint_{model_name}" / "config.json", "w") as f:
            json.dump(hf_config, f, indent=4)
    elif hf_config_src == "path":
        assert hf_config_path is not None, "hf_config_path must be provided if hf_config_src is 'path'"
        shutil.copyfile(hf_config_path, parent_dir / f"hf_checkpoint_{model_name}" / "config.json")
    else:
        raise ValueError("Invalid hf_config_src")

    # optionally, add lit config to hf config for later reference
    if embed_lit_config:
        lit_config = None
        hf_config = None
        with open(parent_dir / f"lit_checkpoint_{model_name}" / "lit_config.json", "r") as f:
            lit_config = json.load(f)
        with open(parent_dir / f"hf_checkpoint_{model_name}" / "config.json", "r") as f:
            hf_config = json.load(f)
        hf_config["lit_config"] = lit_config
        with open(parent_dir / f"hf_checkpoint_{model_name}" / "config.json", "w") as f:
            json.dump(hf_config, f, indent=4)

    ### push to hub
    tokenizer = AutoTokenizer.from_pretrained(parent_dir / f"hf_checkpoint_{model_name}")
    state_dict = torch.load(parent_dir / f"hf_checkpoint_{model_name}/pytorch_model.bin", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(parent_dir / f"hf_checkpoint_{model_name}", state_dict=state_dict)
    model.save_pretrained(
        parent_dir / f"hf_checkpoint_{model_name}", safe_serialization=True
    )  # create the safetensors form

    # above dry runs, below actually pushes to hub
    if not push_to_hub:
        return

    repo_name = f"tomg-group-umd/{model_name}"
    create_repo(repo_name, private=True, token=hf_token)
    model.push_to_hub(repo_name, use_temp_dir=True, token=hf_token)
    tokenizer.push_to_hub(repo_name, use_temp_dir=True, token=hf_token)

    print(f"Model pushed to {repo_name}")


if __name__ == "__main__":
    CLI(convert_checkpoint_to_hf)
