# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional
import warnings

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from litgpt import GPT, Config, PromptStyle, Tokenizer
from litgpt import Config, Tokenizer
from litgpt.model import GPT

# from litgpt.generate.base import generate
from litgpt.generate_base import generate

# from litgpt.prompts import has_prompt_style, load_prompt_style
from litgpt.utils import (
    check_valid_checkpoint_dir,
    # extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

from litgpt.utils import CLI


@torch.no_grad()
def run_generation(
    checkpoint_dir: Path,
    prompt: str = "What food do llamas eat?",
    input: str = "",
    use_chat_template: bool = False,
    finetuned_path: Path = Path("out/full/alpaca/lit_model_finetuned.pth"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,  # strategy defaults to greedy
    top_p: float = 0.0,  # strategy defaults to greedy
    top_k: Optional[int] = None,  # strategy defaults to greedy
    precision: Optional[str] = "32-true",
) -> None:
    """For models finetuned with `litgpt finetune_full`.

    Generates a response based on a given instruction and an optional input. This script will only work with
    checkpoints from the instruction-tuned model. See ``litgpt.finetune.full``.

    Args:
        checkpoint_dir: The path to the checkpoint folder with pretrained model weights.
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            ``litgpt.finetune.full``.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/litgpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        top_p: If specified, it represents the cumulative probability threshold to consider in the sampling process.
            In top-p sampling, the next token is sampled from the highest probability tokens
            whose cumulative probability exceeds the threshold `top_p`. When specified,
            it must be `0 <= top_p <= 1`. Here, `top_p=0` is equivalent
            to sampling the most probable token, while `top_p=1` samples from the whole distribution.
            It can be used in conjunction with `top_k` and `temperature` with the following order
            of application:

            1. `top_k` sampling
            2. `temperature` scaling
            3. `top_p` sampling

            For more details, see https://arxiv.org/abs/1904.09751
            or https://huyenchip.com/2024/01/16/sampling.html#top_p
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    # checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. " "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)
    # config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config = Config.from_file(checkpoint_dir / "lit_config.json")

    config.structured_init = False
    config.structured_init_for_wte = False
    config.structured_init_olmo_variant = False

    checkpoint_path = finetuned_path

    tokenizer = Tokenizer(checkpoint_dir)
    # prompt_style = (
    #     load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir) else PromptStyle.from_config(config)
    # )

    # prompt = prompt_style.apply(prompt, input=input)

    # apply chat template or not
    if use_chat_template:

        def apply_conv_template(input_text, tokenizer):
            message = [
                {"role": "user", "content": input_text},
            ]
            return tokenizer.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        prompt = apply_conv_template(prompt, tokenizer)

    # encoded = tokenizer.encode(prompt, device=fabric.device)
    encoded = tokenizer.encode(prompt, device=fabric.device, bos=True)
    # if both first tokens are bos, remove one of them
    if encoded[0] == tokenizer.bos_id and encoded[1] == tokenizer.bos_id:
        encoded = encoded[1:]

    print(encoded)

    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        config.strategy = "ddp"
        # TODO change to other kernel?? Flash -> Math
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.setup(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    # input_pos = torch.tensor([prompt_length], device=model.device)
    # hstates = model(
    #     encoded.view(1, -1), position_ids=torch.arange(0, prompt_length, device=model.device), return_logits=True
    # )["outputs"]
    # breakpoint()

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(
        model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    # output = output.split("### Response:")[1].strip()
    fabric.print(output)

    # tokens_generated = y.size(0) - prompt_length
    # fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    # if fabric.device.type == "cuda":
    #     fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    CLI(run_generation)
