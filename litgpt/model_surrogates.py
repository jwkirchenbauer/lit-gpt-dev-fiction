# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import os
import math
from typing import Any, Optional, Tuple
from typing_extensions import Self
from functools import partial

from copy import copy

import torch
import torch.nn as nn

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("Torch SDPA settings in model_surrogates.py:")
        print("torch.backends.cuda.flash_sdp_enabled(): ", torch.backends.cuda.flash_sdp_enabled())
        print("torch.backends.cuda.mem_efficient_sdp_enabled(): ", torch.backends.cuda.mem_efficient_sdp_enabled())
        print("torch.backends.cuda.math_sdp_enabled(): ", torch.backends.cuda.math_sdp_enabled())
except Exception as e:
    print(
        f"Failed to import SDPA utils from torch.nn.attention. You probably need a newer version of pytorch for this."
    )
    raise e


from litgpt.config import Config
from litgpt.init import init_normal, scaled_init_normal


def get_linear(config):
    from axonn.intra_layer import Linear as TensorParallelLinear

    if config.strategy == "axonn_tp" and not config.simple_ops:
        return TensorParallelLinear
    else:
        return Linear


# This generic copy.copy pattern in the following getter fns is a bit yucky. Since dataclasses are sort
# of meant to be immutable, but we've dynamically attached things to them, we need to copy them
# to avoid losing on-the-fly attributes added that are not present in the original Config dataclass.


def get_embedding(config):
    embed_kwargs = config._embed_spec.copy()
    embed_config = copy(config)
    for k, v in embed_kwargs.items():
        setattr(embed_config, k, v)

    return nn.Embedding(embed_config.padded_vocab_size, embed_config.n_embd)


def get_lm_head(config):
    lm_head_kwargs = config._lm_head_spec.copy()
    lm_head_config = copy(config)
    for k, v in lm_head_kwargs.items():
        setattr(lm_head_config, k, v)

    return get_linear(lm_head_config)(
        lm_head_config.n_embd, lm_head_config.padded_vocab_size, bias=lm_head_config.lm_head_bias
    )


def get_block_list(config):

    def get_block(block_type):
        block_registry = {"Block": Block, "SqueezeBlock": SqueezeBlock}
        if block_type not in block_registry:
            raise ValueError(f"Block type '{block_type}' not found in the registry")
        return block_registry[block_type]

    block_list = []
    for l in range(config.n_layer):
        block_kwargs = config._block_spec[l].copy()
        block_cls = get_block(block_kwargs.pop("block_type"))

        block_config = copy(config)
        for k, v in block_kwargs.items():
            setattr(block_config, k, v)

        if "head_size" in block_kwargs:
            # we need to override the rope_n_elem if the head_size is overridden
            # since it is derivative from the head_size
            block_config.rope_n_elem = int(block_config.rotary_percentage * block_config.head_size)

        if block_config != config:
            print(f"Block config overridden with: {block_kwargs}")

        block_list.append(block_cls(block_config, l))

    return block_list


class GPT(nn.Module):
    def __init__(
        self,
        config: Config,
        objective: dict[str, Any] = dict(op=torch.nn.functional.cross_entropy, ignore_index=-1),
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()

        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Using model_surrogates.GPT implementation!")

        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=get_embedding(config),
                h=nn.ModuleList(get_block_list(config)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.lm_head = get_lm_head(config)

        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        for block in self.transformer.h:
            block.max_seq_length = block.config.block_size

        self.objective = objective
        self.objective_fn = partial(
            objective["op"],
            ignore_indices=[objective["ignore_index"]],
            gl_strategy=objective["gl_strategy"],
            goldfish_loss_k=objective["goldfish_loss_k"],
            gl_start_position=objective["gl_start_position"],
            gl_context_width=objective["gl_context_width"],
        )
        self.gradient_checkpointing = gradient_checkpointing

    # @property
    # def max_seq_length(self) -> int:
    #     return self._max_seq_length

    # @max_seq_length.setter
    # def max_seq_length(self, value: int) -> None:
    #     """
    #     When doing inference, the sequences used might be shorter than the model's context length.
    #     This allows setting a smaller number to avoid allocating unused memory
    #     """
    #     if value > self.config.block_size:
    #         raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
    #     self._max_seq_length = value
    #     if not hasattr(self, "cos"):
    #         # first call
    #         cos, sin = self.rope_cache()
    #         self.register_buffer("cos", cos, persistent=False)
    #         self.register_buffer("sin", sin, persistent=False)
    #     # override
    #     elif value != self.cos.size(0):
    #         self.cos, self.sin = self.rope_cache(device=self.cos.device)
    #     # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
    #     # if the kv cache is expected

    # def reset_parameters(self) -> None:
    #     # Trigger resetting the rope-cache
    #     self.cos, self.sin = self.rope_cache(device=self.cos.device)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        T = idx.size(1)
        assert attention_mask is None
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if position_ids is not None:  # use the kv cache
            # cos = self.cos.index_select(0, position_ids)
            # sin = self.sin.index_select(0, position_ids)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, position_ids)
        else:
            # cos = self.cos[:T]
            # sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        for block in self.transformer.h:
            if position_ids is not None:  # use the kv cache
                cos = block.cos.index_select(0, position_ids)
                sin = block.sin.index_select(0, position_ids)
            else:
                cos = block.cos[:T]
                sin = block.sin[:T]
            if not self.gradient_checkpointing:
                x = block(x, cos, sin, mask, position_ids)
            else:
                x = self.config.checkpoint(block, x, cos, sin, mask, position_ids)
        x = self.transformer.ln_f(x)
        outputs = self.lm_head(x)
        loss = 0.0
        if labels is not None:
            loss = self.objective_fn(outputs, labels, training=self.training)
        return {"loss": loss, "outputs": outputs if return_logits else None}

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    # def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return build_rope_cache(
    #         seq_len=self.max_seq_length,
    #         n_elem=self.config.rope_n_elem,
    #         device=device,
    #         condense_ratio=self.config.rope_condense_ratio,
    #         base=self.config.rope_base,
    #     )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # if rope_cache_length is None:
        #     rope_cache_length = self.cos.size(-1)
        assert rope_cache_length is None, "block local logic requires not passing `rope_cache_length` to `set_kv_cache`"
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            rope_cache_length = block.cos.size(-1)
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None


class Block(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config, layer_idx)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓             ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───►  norm_2
           │  ↓                     │  ↓             ↓
           │  attn                  │  attn          mlp
           │  ↓                     │  ↓             │
        ┌─ └► +                     └► + ◄───────────┘
        │     norm_2
        │     ↓
        │     mlp
        │     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )


class SqueezeBlock(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config, layer_idx)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        if self.config.parallel_residual:
            raise NotImplementedError("SqueezeBlock does not support parallel_residual=True")
        else:
            x = attention_output + x
            # x = self.mlp(self.norm_2(x)) + x
            if isinstance(self.mlp, AsymmetricLLaMAMLP):
                # Since the asymm MLP resizes the tensors, we can't add another copy of the input
                # so this style means there's a single residual connection in this particular block.
                x = self.mlp(self.norm_2(x))
            else:
                x = self.mlp(self.norm_2(x)) + x
        return x

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = get_linear(config)(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = get_linear(config)(
            config.head_size * config.n_head,
            config.n_embd,
            bias=config.bias,
            init_method=scaled_init_normal(config.n_embd, layer_idx),
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)

        # Switch between different SDPA backends, hardcoded for now.
        # torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True # helps debug why the kernel failed to run.
        # with sdpa_kernel(SDPBackend.MATH):
        # with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        self.fc = get_linear(config)(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = get_linear(config)(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            init_method=scaled_init_normal(config.n_embd, layer_idx),
        )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        self.fc_1 = get_linear(config)(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = get_linear(config)(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = get_linear(config)(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            init_method=scaled_init_normal(config.n_embd, layer_idx),
        )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class AsymmetricLLaMAMLP(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        self.fc_1 = get_linear(config)(config.n_embd_mlp_in, config.intermediate_size, bias=config.bias)
        self.fc_2 = get_linear(config)(config.n_embd_mlp_in, config.intermediate_size, bias=config.bias)
        self.proj = get_linear(config)(
            config.intermediate_size,
            config.n_embd_mlp_out,
            bias=config.bias,
            init_method=scaled_init_normal(config.n_embd_mlp_out, layer_idx),
        )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = get_linear(config)(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        if self.add_unit_offset:
            # Gemma model requires a unit offset
            # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
            return x_normed * (1 + self.weight)
        return x_normed * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class Linear(torch.nn.Linear):
    """Linear layer wrapper that unifies tensor-parallel implementation and default implementations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        init_method=None,
    ):
        self.init_method = init_method if init_method else init_normal(in_features)
        super().__init__(in_features, out_features, bias, device, dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, **kwargs):
        return super().forward(x)
