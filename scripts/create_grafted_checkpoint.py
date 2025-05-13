import os
import gc
import shutil
import json
from pathlib import Path
from typing import Union

import torch.nn as nn

from litgpt.utils import find_multiple
from litgpt.utils import incremental_save, lazy_load
from litgpt.utils import CLI
from litgpt.model import GPT
from scripts.convert_hf_checkpoint import load_param


def copy_and_graft(state_dict, lit_weights_to, lit_weights_from, saver, params_to_graft=[], coerce_dtype=False):

    for name, param in lit_weights_to.items():

        param = load_param(param, name, None)  # although this is namedparam it actually returns a torch tensor

        if any([p_name in name for p_name in params_to_graft]):

            param_from = lit_weights_from.get(name, None)
            assert param_from is not None, f"Could not find {name} in the model to graft from."
            param_from = load_param(param_from, name, None)
            assert param_from.shape == param.shape, f"Shape mismatch for {name}."
            if not coerce_dtype:
                assert param_from.dtype == param.dtype, f"Dtype mismatch for {name}."
            else:
                param_from = param_from.to(param.dtype)

            param = param_from

            print(f"Grafted {name} from donor model.")

        if saver is not None:
            param = saver.store_early(param)

        state_dict[name] = param


def create_grafted_checkpoint(
    base_checkpoint_dir: Path,  # Path to the checkpoint dir for the model to receive the graft
    embedding_checkpoint_dir: Path,  # Path to the checkpoint dir for the model to donate the embeddings
    params_to_graft: Union[str, list] = ["lm_head.weight", "transformer.wte.weight"],  # List of parameters to graft
    coerce_dtype: bool = False,  # Coerce the dtype of the embeddings to match the base model
):
    """Grafts the embeddings and lm_head from one model to another."""

    if isinstance(params_to_graft, str):
        params_to_graft = json.loads(params_to_graft)
    print(f"params_to_graft: {params_to_graft}")

    base_cfg_file = base_checkpoint_dir / "lit_config.json"
    embed_cfg_file = embedding_checkpoint_dir / "lit_config.json"

    with open(base_cfg_file) as f:
        base_model_config = json.load(f)
    with open(embed_cfg_file) as f:
        embed_model_config = json.load(f)

    assert base_model_config["lm_head_bias"] == False, "This script does not support lm_head_bias currently, but could."
    assert (
        embed_model_config["lm_head_bias"] == False
    ), "This script does not support lm_head_bias currently, but could."

    assert base_model_config["vocab_size"] == embed_model_config["vocab_size"], "Vocab sizes must match."
    assert (
        base_model_config["padded_vocab_size"] == embed_model_config["padded_vocab_size"]
    ), "Padded vocab sizes must match."

    # this only works for llama models rn
    assert (
        "llama" in base_model_config["name"].lower() and "llama" in embed_model_config["name"].lower()
    ), "This script only works for llama models currently."

    # perform the graft
    try:
        os.remove(base_checkpoint_dir / "lit_model_out.pth")
    except FileNotFoundError:
        pass

    sd = {}
    with incremental_save(str(base_checkpoint_dir / "lit_model_out.pth")) as saver:
        lit_weights_to = lazy_load(str(base_checkpoint_dir / "lit_model.pth"))
        lit_weights_to = lit_weights_to.get("model", lit_weights_to)
        lit_weights_from = lazy_load(str(embedding_checkpoint_dir / "lit_model.pth"))
        lit_weights_from = lit_weights_from.get("model", lit_weights_from)

        copy_and_graft(
            sd,
            lit_weights_to,
            lit_weights_from,
            saver=saver,
            params_to_graft=params_to_graft,
            coerce_dtype=coerce_dtype,
        )
        gc.collect()
        saver.save(sd)

    os.remove(base_checkpoint_dir / "lit_model.pth")
    os.rename(base_checkpoint_dir / "lit_model_out.pth", base_checkpoint_dir / "lit_model.pth")


if __name__ == "__main__":
    CLI(create_grafted_checkpoint)
