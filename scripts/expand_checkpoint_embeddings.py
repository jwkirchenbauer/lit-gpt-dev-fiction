import os
import gc
import shutil
import json
from pathlib import Path

import torch.nn as nn

from litgpt.utils import find_multiple
from litgpt.utils import incremental_save, lazy_load
from litgpt.utils import CLI
from litgpt.model import GPT

from litgpt.expand_embeddings import copy_and_expand


def expand_checkpoint_embeds(
    checkpoint_dir: Path,  # Path to the checkpoint dir
    target_size: int,  # Target size of the vocab
    zero_useless_rows: bool = True,  # Zero out the rows that are not needed
):
    """Expands the embeddings of a litgpt checkpoint to a new size based on config file.

    Warning: This is currently written to be a fully in place operation. Make sure to be
    careful with your checkpoint directories. If you want to keep the old checkpoint, make
    a copy of it first and target the copy dir.
    """

    cfg_file = checkpoint_dir / "lit_config.json"

    with open(cfg_file) as f:
        model_config = json.load(f)

    assert model_config["lm_head_bias"] == False, "This script does not support lm_head_bias currently, but could."

    if model_config["vocab_size"] != target_size:
        print(f"Expanding embeddings from {model_config['vocab_size']} to {target_size}")
        model_config["vocab_size"] = target_size
    else:
        print("Config vocab_size is already the same as target_size. This could be a no-op, continuing...")

    padded_target_size = find_multiple(model_config["vocab_size"], model_config["padding_multiple"])
    if padded_target_size != model_config["padded_vocab_size"]:
        model_config["padded_vocab_size"] = padded_target_size
    else:
        print("No need to actually expand, we have enough rows in the embeddings already. Still continuing...")

    with open(cfg_file, "w") as f:
        json.dump(model_config, f, indent=4)

    sd = {}
    with incremental_save(str(checkpoint_dir / "lit_model_out.pth")) as saver:
        lit_weights = lazy_load(str(checkpoint_dir / "lit_model.pth"))
        lit_weights = lit_weights.get("model", lit_weights)
        copy_and_expand(
            sd,
            lit_weights,
            saver=saver,
            new_size=padded_target_size,
            new_vocab_size=target_size,
            zero_useless_rows=zero_useless_rows,
        )
        gc.collect()
        saver.save(sd)

    os.remove(checkpoint_dir / "lit_model.pth")
    os.rename(checkpoint_dir / "lit_model_out.pth", checkpoint_dir / "lit_model.pth")


if __name__ == "__main__":
    CLI(expand_checkpoint_embeds)
