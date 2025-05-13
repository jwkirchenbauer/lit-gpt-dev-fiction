# type: ignore
from copy import deepcopy

"""This config file contains only working / in-progress architecture definitions for model_dynamic.py
Refer to config.py for static litgpt definitions that work with model.py
"""

configs = []

###############
# Baselines
###############

baselines = [
    dict(
        name="baby-llama-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=32768,
        n_layer=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=1024,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=2816,
        init_strategy="scaled",
    ),
    dict(
        name="baby-llama-long-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-long-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=24,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=768,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=2048,
        init_strategy="scaled",
    ),
    dict(
        name="baby-snake-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-long-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=48,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=512,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=1536,
        init_strategy="scaled",
    ),
    dict(
        name="gpt2-124m",
        hf_config=dict(org="tomg-group-umd", name="gpt2-124m"),
        block_size=1024,
        vocab_size=32000,  # actually 50k but we aren't retokenizing for that
        padding_multiple=4096,
        n_layer=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        n_embd=768,
        bias=False,
        norm_class_name="LayerNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="GELU",
        intermediate_size=3072,
        init_strategy="scaled",
    ),
]
configs.extend(baselines)

###############
# Sanity Checks
###############

sanity = [
    dict(
        name="brick-200m",
        hf_config=dict(org="tomg-group-umd", name="brick-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=4,
        n_embd=2048,
        bias=False,
        norm_class_name="Identity",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU",
        intermediate_size=8192,
        init_strategy="normal",
        attn_impl="debug-skip",
    ),
    dict(
        name="big-brick-200m",
        hf_config=dict(org="tomg-group-umd", name="big-brick-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=24,
        n_embd=4096,
        bias=False,
        block_class_name="Brick",
        mlp_class_name="BrickLP",
        nonlin_name="ReLU",
        intermediate_size=4096,
        init_strategy="normal",
        attn_impl="debug-skip",
    ),
    #     Iteration   1024 | Loss: 63.4014 | 3426761036166133342386782208.00 PPL      | Update      8|
    #  (optimizer.step)| MFU : 78.13%  | tok/sec: 144291.1 | steps/sec: 4.40 |
]
configs.extend(sanity)


###############
# Meta LLaMA 2
###############
llama_2 = [
    dict(
        name="Llama-debug",
        hf_config=dict(org="tomg-group-umd", name="Llama-debug"),
        vocab_size=32000,
        block_size=384,  # to find it again
        num_attention_heads=8,
        num_key_value_heads=2,
        n_embd=512,
        padding_multiple=4096,
        n_layer=4,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=256,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


# ###############
# # Meta LLaMA 3
# ###############
# llama_3 = [
#     # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
#     dict(
#         name="Llama-3-8B{}",
#         hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
#         block_size=8192,
#         vocab_size=128256,
#         padding_multiple=4096,
#         n_layer=32,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=14336,
#         rope_base=500000,
#     ),
#     # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
#     dict(
#         name="Llama-3-70B{}",
#         hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
#         block_size=8192,
#         vocab_size=128256,
#         padding_multiple=4096,
#         n_layer=80,
#         num_attention_heads=64,
#         n_embd=8192,
#         num_key_value_heads=8,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=28672,
#         rope_base=500000,
#     ),
# ]
# for c in llama_3:
#     for kind in ("", "-Instruct"):
#         copy = deepcopy(c)
#         copy["name"] = c["name"].format(kind)
#         copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
#         configs.append(copy)


# ##################################
# # togethercomputer LLaMA-2-7B-32K
# ##################################
# together_llama2_32k = [
#     # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
#     dict(
#         name="LLaMA-2-7B-32K",
#         hf_config=dict(org="togethercomputer", name="LLaMA-2-7B-32K"),
#         vocab_size=32000,
#         padding_multiple=4096,
#         n_layer=32,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=11008,
#         rope_condense_ratio=8,
#     )
# ]
# configs.extend(together_llama2_32k)


# ################
# # Microsoft Phi
# ################
# phi = [
#     # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
#     dict(
#         name="phi-1_5",
#         hf_config=dict(org="microsoft", name="phi-1_5"),
#         vocab_size=50257,
#         padded_vocab_size=51200,
#         block_size=2048,
#         n_embd=2048,
#         n_layer=24,
#         rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
#         shared_attention_norm=True,
#         lm_head_bias=True,
#         gelu_approximate="tanh",
#     ),
#     # https://huggingface.co/microsoft/phi-2/blob/main/config.json
#     dict(
#         name="phi-2",
#         hf_config=dict(org="microsoft", name="phi-2"),
#         vocab_size=50257,
#         padded_vocab_size=51200,
#         block_size=2048,
#         n_embd=2560,
#         n_layer=32,
#         rotary_percentage=0.4,  # 32 / (n_embd / n_head) = 32 / 80
#         shared_attention_norm=True,
#         lm_head_bias=True,
#         gelu_approximate="tanh",
#     ),
# ]
# configs.extend(phi)


# ############
# # TinyLlama
# ############
# tiny_llama = [
#     dict(
#         name="tiny-llama-190m",
#         hf_config=dict(org="tomg-group-umd", name="tiny-llama-190m"),
#         block_size=2048,
#         vocab_size=32000,
#         padding_multiple=4096,
#         n_layer=12,
#         num_attention_heads=32,
#         n_embd=1024,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",  # original TinyLlama uses FusedRMSNorm
#         norm_eps=1e-5,
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=2816,
#         num_key_value_heads=4,
#     ),
#     dict(
#         name="tiny-llama-1.1b{}",
#         hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
#         block_size=2048,
#         vocab_size=32000,
#         padding_multiple=4096,
#         n_layer=22,
#         num_attention_heads=32,
#         n_embd=2048,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",  # original TinyLlama uses FusedRMSNorm
#         norm_eps=1e-5,
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=5632,
#         num_key_value_heads=4,
#     ),
# ]
# for c in tiny_llama:
#     for kind, hf_postfix in (("", "-intermediate-step-1431k-3T"), ("-chat", "-Chat-v1.0")):
#         copy = deepcopy(c)
#         copy["name"] = c["name"].format(kind)
#         copy["hf_config"]["name"] = c["hf_config"]["name"].format(hf_postfix)
#         configs.append(copy)


name_to_config = {config["name"]: config for config in configs}
