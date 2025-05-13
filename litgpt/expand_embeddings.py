import os
import gc
import shutil


import torch.nn as nn

from litgpt.utils import find_multiple
from litgpt.utils import incremental_save, lazy_load

from litgpt.model import GPT
from scripts.convert_hf_checkpoint import load_param


def expand_embedding_weight(param, new_size, new_vocab_size, zero_useless_rows=False):
    # https://github.com/huggingface/transformers/blob/02300273e220932a449a47ebbe453e7789be454b/src/transformers/modeling_utils.py#L1990
    old_size = param.shape[0]

    if old_size == new_size:
        print("Since old_size and new_size are the same, the embeddings are already the correct size. This is a No-op.")

    new_emb_layer = nn.Embedding(new_size, param.shape[1], device=param.device, dtype=param.dtype)
    GPT._init_weights(new_emb_layer)
    new_tensor = new_emb_layer.weight.data

    new_tensor[:old_size, :] = param[:old_size, :]
    if zero_useless_rows:
        new_tensor[new_vocab_size:, :] = 0  # zero the uneeded rows
    return new_tensor


def expand_lm_head(param, new_size, new_vocab_size, bias=False, zero_useless_rows=False):
    # https://github.com/huggingface/transformers/blob/02300273e220932a449a47ebbe453e7789be454b/src/transformers/modeling_utils.py#L2094
    if bias:
        raise NotImplementedError("This function does not support bias in the lm_head yet.")

    ## then must be main weight
    old_size = param.shape[0]

    if old_size == new_size:
        print("Since old_size and new_size are the same, the lm_head is already the correct size. This is a No-op.")

    new_lm_head = nn.Linear(param.shape[1], new_size, bias=bias, device=param.device, dtype=param.dtype)
    GPT._init_weights(new_lm_head)
    new_tensor = new_lm_head.weight.data

    new_tensor[:old_size, :] = param[:old_size, :]
    if zero_useless_rows:
        new_tensor[new_vocab_size:, :] = 0  # zero the uneeded rows
    return new_tensor


def copy_and_expand(state_dict, lit_weights, saver, new_size, new_vocab_size, zero_useless_rows=False):
    for name, param in lit_weights.items():
        param = load_param(param, name, None)  # although this is namedparam it actually returns a torch tensor
        if "lm_head.weight" in name:
            param = expand_lm_head(
                param, new_size=new_size, new_vocab_size=new_vocab_size, zero_useless_rows=zero_useless_rows
            )
        if "transformer.wte.weight" in name:
            param = expand_embedding_weight(
                param, new_size=new_size, new_vocab_size=new_vocab_size, zero_useless_rows=zero_useless_rows
            )
        if saver is not None:
            param = saver.store_early(param)

        state_dict[name] = param


def expand_embeddings_for_delete(new_size, cfg, output_path=None, pad_matracies=True):
    if cfg.model_checkpoint is None:
        return cfg  # training from scratch so embeddings can be any shape we decide

    cfg.model_config.vocab_size = new_size

    if pad_matracies:
        padded_vocab_size_with_delete_token = find_multiple(
            cfg.model_config.vocab_size, cfg.model_config.padding_multiple
        )
    else:
        padded_vocab_size_with_delete_token = cfg.model_config.vocab_size
    if padded_vocab_size_with_delete_token != cfg.model_config.padded_vocab_size:
        cfg.model_config.padded_vocab_size = padded_vocab_size_with_delete_token
    else:
        return cfg  # no need to expand, we had the spare space in the checkpoint anyway

    checkpoint_path = cfg.model_checkpoint

    if output_path is None:
        parts = checkpoint_path.split("/external/")
        output_path = (
            "/lustre/orion/csc569/scratch/smcleish/delete_expanded_models/"
            + parts[1]
            + f"_expanded_embeddings_{cfg.model_config.padded_vocab_size}"
        )

    # check if path already exists
    if os.path.exists(output_path):
        cfg.model_checkpoint = output_path
        return cfg
    else:
        print("Making: ", output_path)
        shutil.copytree(checkpoint_path, output_path, dirs_exist_ok=True)
        os.chmod(output_path, 0o2775)

    sd = {}
    with incremental_save(output_path + "/lit_model.pth") as saver:
        lit_weights = lazy_load(checkpoint_path + "/lit_model.pth")
        lit_weights = lit_weights.get("model", lit_weights)
        copy_and_expand(
            sd,
            lit_weights,
            saver=saver,
            new_size=cfg.model_config.padded_vocab_size,
            new_vocab_size=cfg.model_config.vocab_size,
            zero_useless_rows=True,
        )
        gc.collect()
        saver.save(sd)

    cfg.model_checkpoint = output_path
    return cfg  # with new model name in
