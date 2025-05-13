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

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

# The following imports are not available if torch is old, eg. torch 2.2
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception as e:
    SDPBackend = None
    sdpa_kernel = None
    import contextlib

    null_ctx = contextlib.nullcontext()

# Force the SDPA kernel...
# in a pre ctx manager version friendly way ...
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

# and post ctx manager version way.
if sdpa_kernel is not None and SDPBackend is not None:
    SDP_BACKEND_CHOICE = SDPBackend.FLASH_ATTENTION
    # SDP_BACKEND_CHOICE = SDPBackend.EFFICIENT_ATTENTION
    # SDP_BACKEND_CHOICE = SDPBackend.MATH

# Extra flag that helps debug why the specific kernel failed to run.
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None

from litgpt.config import Config
from litgpt.init import init_normal, scaled_init_normal, structured_init_normal

from .ops import LinearCrossEntropyLoss


def get_linear(config, use_standard_linear=False):
    assert not use_standard_linear, "what was use_standard_linear arg for? it doesnt do anything"
    from axonn.intra_layer import Linear as TensorParallelLinear

    if config.strategy == "axonn_tp" and not config.simple_ops:
        return TensorParallelLinear
    else:
        return Linear


class GPT(nn.Module):
    def __init__(
        self,
        config: Config,
        objective: dict[str, Any] = dict(op=torch.nn.functional.cross_entropy, ignore_index=-1),
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()

        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Using model.GPT implementation!")

        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=Embedding(
                    config.padded_vocab_size,
                    config.n_embd,
                    init_method=(
                        None
                        if not (self.config.structured_init_for_wte or self.config.structured_init_olmo_variant)
                        else structured_init_normal(
                            config.n_embd,
                            config.n_layer,
                            weight_type="wte",
                            use_olmo_variant=self.config.structured_init_olmo_variant,
                        )
                    ),
                ),
                h=nn.ModuleList(Block(config, l) for l in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )

        self.lm_head = None

        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

        self.objective = objective
        self.set_objective()

        self.gradient_checkpointing = gradient_checkpointing
        self.monitoring = False
        self.latest_metrics = {}

    def set_objective(self, objective: dict[str, Any] = None) -> None:
        if objective is None:
            assert hasattr(self, "objective"), "No objective set"
            objective = self.objective
        else:
            self.objective = objective

        lm_head_exists = self.lm_head is not None
        if lm_head_exists:
            current_is_transposed = isinstance(self.lm_head, LinearCrossEntropyLoss) and self.lm_head.transposed_weight
            current_lm_head_weights = self.lm_head.weight.data
            current_lm_head_bias = self.lm_head.bias.data if self.lm_head.bias is not None else None
            del self.lm_head
        else:
            current_lm_head_weights = None
            current_lm_head_bias = None

        if objective.get("use_jonas_ce"):
            assert not self.config.lm_head_bias, "We don't support bias in the LM head with LinearCrossEntropyLoss impl"
            self.lm_head = LinearCrossEntropyLoss(
                self.config.n_embd,
                self.config.padded_vocab_size,
                ignore_index=self.objective.get("ignore_index"),
                # z_regularization=objective["z_regularization"], # not supported
                logit_scale=self.config.init.logit_scale,  # not using these
                init_method=self.config.init.fn("head"),  # not using these
                # transposed_weight=True,  # default ? Not tying, not supported?
            )
            print(f"Creating Jonas CE with init {self.config.init}")
            if current_lm_head_weights is not None:
                self.lm_head.weight.data = (
                    # current_lm_head_weights if not current_is_transposed else current_lm_head_weights.T
                    current_lm_head_weights.T
                    if not current_is_transposed
                    else current_lm_head_weights
                )
            if current_lm_head_bias is not None:
                raise ValueError("Cannot transfer bias to the LinearCrossEntropyLoss LM head")
        else:
            # self.lm_head = get_linear(self.config)(
            self.lm_head = get_linear(self.config, use_standard_linear=objective.get("use_liger_ce"))(
                self.config.n_embd,
                self.config.padded_vocab_size,
                bias=self.config.lm_head_bias,
                init_method=(
                    None
                    if not (self.config.structured_init or self.config.structured_init_olmo_variant)
                    else structured_init_normal(
                        self.config.n_embd,
                        self.config.n_layer,
                        weight_type="lm_head",
                        use_olmo_variant=self.config.structured_init_olmo_variant,
                    )
                ),
            )
            if current_lm_head_weights is not None:
                self.lm_head.weight.data = (
                    # current_lm_head_weights if not current_is_transposed else current_lm_head_weights.T
                    current_lm_head_weights.T
                    if current_is_transposed
                    else current_lm_head_weights
                )
            if current_lm_head_bias is not None:
                self.lm_head.bias.data = current_lm_head_bias

        # Ideally, we hard require only the loss operation but it is defaulted on construction,
        # and everything else is optional
        # args hopefully handled by the loss function itself (in utils.py probably)
        self.objective_fn = partial(
            self.objective["op"],
            ignore_indices=[self.objective.get("ignore_index")],
            gl_k=self.objective.get("gl_k"),
            gl_strategy=self.objective.get("gl_strategy"),
            gl_start_position=self.objective.get("gl_start_position"),
            gl_context_width=self.objective.get("gl_context_width"),
            target_range=self.objective.get("target_range"),
            return_logits_targets=self.objective.get("return_logits_targets"),
        )

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
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        use_liger_ce: bool = False,
        reduction="mean",
    ) -> dict[str, Optional[torch.Tensor]]:
        T = input_ids.size(1)
        assert attention_mask is None
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if position_ids is not None:  # use the kv cache
            cos = self.cos.index_select(0, position_ids)
            sin = self.sin.index_select(0, position_ids)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, position_ids)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        for block in self.transformer.h:
            if not self.gradient_checkpointing:
                x = block(x, cos, sin, mask, position_ids)
            else:
                x = self.config.checkpoint(block, x, cos, sin, mask, position_ids)
        x = self.transformer.ln_f(x)
        clamp_head = (
            partial(do_softcapping, thresh=self.config.final_logit_softcapping)
            if self.config.final_logit_softcapping is not None
            else nn.Identity()
        )

        if use_liger_ce:
            assert labels is not None, "Labels are required for LIGER CE"
            assert not return_logits, "Cannot return logits when using LIGER CE"
            assert not self.objective.get("return_logits_targets"), "Cannot return logits when using LIGER CE"
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

            shift_hidden_states = x[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_hidden_states = shift_hidden_states.view(-1, self.config.n_embd)
            shift_labels = shift_labels.view(-1)

            lce = LigerFusedLinearCrossEntropyLoss()
            loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
            return {"loss": loss}
        elif self.objective.get("use_jonas_ce"):
            assert labels is not None, "Labels are required for Jonas CE"
            assert not return_logits, "Cannot return logits when using Jonas CE"
            assert not self.objective.get("return_logits_targets"), "Cannot return logits when using Jonas CE"

            print(f"Using Jonas CE {type(self.lm_head)}")

            loss = clamp_head(self.lm_head(x, labels))
            return {"loss": loss}

        outputs = clamp_head(self.lm_head(x))
        if self.monitoring:
            self.monitor_module(x, outputs)
        if labels is not None:
            loss = self.objective_fn(outputs, labels, training=self.training, reduction=reduction)
            if self.objective.get("return_logits_targets"):
                # the loss is a tuple with three elements that we will unpack in other code
                # please FIXME this is terrible in context of the lines below
                return {"loss": loss}
        else:
            loss = torch.as_tensor(0.0)
        return {"loss": loss, "logits": outputs if return_logits else None}

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard rope
                extra_config = {name: self.config.rope_adjustments[name] for name in adjusted_params_required}
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param for param, present in zip(adjusted_params_required, params_present) if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
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

    @torch.no_grad()
    def monitor_module(self, x: torch.Tensor, logits: torch.Tensor):
        z_value = torch.logsumexp(logits, dim=-1).pow(2).mean()
        normed_x = x / x.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x.shape[1]
        probs = torch.softmax(logits, dim=-1)
        logit_ent = torch.where(probs > 0, -probs * probs.log(), torch.zeros_like(probs)).sum(dim=-1).mean()
        metrics = {
            "last_hidden_token_corr": token_corr,
            "last_hidden_norm": x.norm(),
            "logit_mean": logits.mean(),
            "logit_norm": logits.norm(),
            "logit_entropy": logit_ent,
            "z_value": z_value,
        }
        self.latest_metrics = metrics  # will be picked up from monitoring caller


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
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = (
            self.norm_1 if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        )
        self.mlp = config.mlp_class(config, layer_idx)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )

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
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)
        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)
        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        attn_init_method = (
            None
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(
                config.n_embd,
                config.n_layer,
                weight_type="attn_qkv",
                use_olmo_variant=config.structured_init_olmo_variant,
            )
        )
        if config.strategy == "axonn_tp":
            self.attn = get_linear(config)(
                config.n_embd,
                shape,
                bias=config.bias or config.attn_bias,
                expert_mode=True,
                init_method=attn_init_method,
            )
        else:
            self.attn = get_linear(config)(
                config.n_embd,
                shape,
                bias=config.bias or config.attn_bias,
                init_method=attn_init_method,
            )
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        olmo_kwargs = dict(use_olmo_variant=config.structured_init_olmo_variant)
        if config.structured_init_olmo_variant:
            olmo_kwargs["layer_idx"] = layer_idx
            olmo_kwargs["head_size"] = config.head_size
            olmo_kwargs["n_head"] = config.n_head

        proj_init_method = (
            scaled_init_normal(config.n_embd, layer_idx)
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(config.n_embd, config.n_layer, weight_type="attn_proj", **olmo_kwargs)
        )
        if config.strategy == "axonn_tp":
            self.proj = get_linear(config)(
                config.head_size * config.n_head,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
                expert_mode=True,
                transpose=True,
            )
        else:
            self.proj = get_linear(config)(
                config.head_size * config.n_head,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
            )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and layer_idx % config.sliding_window_layer_stride == 0
        )

        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size * config.n_head, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size * config.n_query_groups, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config

        self.config.FAST_ATTN_DISABLED = False
        self.config.FAST_ATTN_DISABLED_WARNED_ONCE = False

        if config.strategy == "axonn_tp":
            # adjust number of heads
            from copy import deepcopy
            from axonn import axonn as ax

            self.config = deepcopy(self.config)
            attention_world_size = ax.config.G_intra_r
            assert self.config.n_head % attention_world_size == 0
            self.config.n_head //= attention_world_size
            assert self.config.n_query_groups % attention_world_size == 0
            self.config.n_query_groups //= attention_world_size

        if self.check_rocm_attn_conditions():
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            self.rocm_flash_attention = FlashSelfAttention(
                causal=True, attention_dropout=0, softmax_scale=scale  # hard coded for now
            )
        else:
            self.rocm_flash_attention = None

    def check_rocm_attn_conditions(
        self,
        q_dtype: torch.dtype = torch.bfloat16,
        using_kv_cache: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> bool:
        # validation logic for ROCm Flash Attention
        rocm_attn_chosen = self.config.attn_impl == "rocm"
        dtype_valid_for_rocm_attn = q_dtype in (torch.float16, torch.bfloat16)
        rocm_attn_fn_avail = flash_attn_unpadded_func is not None
        not_using_kv_cache = not using_kv_cache
        attn_mask_is_none = mask is None
        if rocm_attn_chosen:
            assert dtype_valid_for_rocm_attn, "ROCm Flash Attention requires FP16 or BF16 input tensors"
            assert rocm_attn_fn_avail, "ROCm Flash Attention is not available"
            assert not_using_kv_cache, "Cannot use KV cache with ROCm Flash Attention. If running inference use SDPA."
            assert (
                attn_mask_is_none
            ), "ROCm Flash Attention doesn't support a custom attention mask being passed. Use SDPA or Eager."
        rocm_attn_conds = all([rocm_attn_chosen, dtype_valid_for_rocm_attn, rocm_attn_fn_avail, not_using_kv_cache])
        return rocm_attn_conds

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

        # NOTE this may not be perfect since implementation changed upstream since/when was added
        if self.config.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

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

        using_kv_cache = False
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)
            using_kv_cache = True

        if self.apply_sliding_window_attention:
            self.config.FAST_ATTN_DISABLED = True
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
                mask = mask.view(1, 1, *mask.shape)
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        # with softcapping we cannot use SDPA/FA
        if self.config.attention_logit_softcapping is not None:
            self.config.FAST_ATTN_DISABLED = True
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            scores = q @ k.mT * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
            y = y.transpose(1, 2)
        else:
            # execute rocm attn if the conditions are met
            if self.check_rocm_attn_conditions(q_dtype=q.dtype, using_kv_cache=using_kv_cache, mask=mask):
                q = q.permute(0, 2, 1, 3)  # (B, nh, T, hs)
                k = k.permute(0, 2, 1, 3)  # (B, nh, T, hs)
                v = v.permute(0, 2, 1, 3)  # (B, nh, T, hs)

                y = self.rocm_flash_attention(q, k, v)
            else:
                y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        if self.config.FAST_ATTN_DISABLED and (not self.config.FAST_ATTN_DISABLED_WARNED_ONCE):
            print(
                "Model configs for attention (windowed, softcapping) have likely disabled fast attention via rocm or sdpa!"
            )
            self.config.FAST_ATTN_DISABLED_WARNED_ONCE = True

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        if sdpa_kernel is not None and SDPBackend is not None:
            ctx = sdpa_kernel(SDP_BACKEND_CHOICE)
        else:
            ctx = null_ctx
        with ctx:
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
        fc_init_method = (
            None
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(
                config.n_embd,
                config.n_layer,
                weight_type="mlp_fc",
                use_olmo_variant=config.structured_init_olmo_variant,
            )
        )
        olmo_kwargs = dict(use_olmo_variant=config.structured_init_olmo_variant)
        if config.structured_init_olmo_variant:
            olmo_kwargs["layer_idx"] = layer_idx
            olmo_kwargs["intermediate_size"] = config.intermediate_size
        proj_init_method = (
            scaled_init_normal(config.n_embd, layer_idx)
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(config.n_embd, config.n_layer, weight_type="mlp_proj", **olmo_kwargs)
        )
        if config.strategy == "axonn_tp":
            self.fc = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                expert_mode=True,
                init_method=fc_init_method,
            )
            self.proj = get_linear(config)(
                config.intermediate_size,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
                expert_mode=True,
                transpose=True,
            )
        else:
            self.fc = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                init_method=fc_init_method,
            )
            self.proj = get_linear(config)(
                config.intermediate_size,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
            )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config, layer_idx: int = 0) -> None:
        super().__init__()

        fc_init_method = (
            None
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(
                config.n_embd,
                config.n_layer,
                weight_type="mlp_fc",
                use_olmo_variant=config.structured_init_olmo_variant,
            )
        )
        olmo_kwargs = dict(use_olmo_variant=config.structured_init_olmo_variant)
        if config.structured_init_olmo_variant:
            olmo_kwargs["layer_idx"] = layer_idx
            olmo_kwargs["intermediate_size"] = config.intermediate_size
        proj_init_method = (
            scaled_init_normal(config.n_embd, layer_idx)
            if not (config.structured_init or config.structured_init_olmo_variant)
            else structured_init_normal(config.n_embd, config.n_layer, weight_type="mlp_proj", **olmo_kwargs)
        )
        if config.strategy == "axonn_tp":
            self.fc_1 = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                expert_mode=True,
                init_method=fc_init_method,
            )
            self.fc_2 = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                expert_mode=True,
                init_method=fc_init_method,
            )
            self.proj = get_linear(config)(
                config.intermediate_size,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
                expert_mode=True,
                transpose=True,
            )
        else:
            self.fc_1 = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                init_method=fc_init_method,
            )
            self.fc_2 = get_linear(config)(
                config.n_embd,
                config.intermediate_size,
                bias=config.bias,
                init_method=fc_init_method,
            )
            self.proj = get_linear(config)(
                config.intermediate_size,
                config.n_embd,
                bias=config.bias,
                init_method=proj_init_method,
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
        assert config.structured_init is False, "Structured init not added for MoE"
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
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
            Shapes are `(seq_len, n_elem)`.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    # If `n_elem` is odd, the final dimension of `idx_theta` has size
    # `n_elem + 1`, so need to cut something off.
    # Due to a current bug in Hugging Face, in the case `n_elem == 1`, we leave
    # `idx_theta`, `cos`, `sin` as is. Things work out in `apply_rope` due to
    # broadcasting. If we shorten `idx_theta`, unit tests comparing to
    # Hugging Face fail.
    # https://github.com/huggingface/transformers/issues/35233
    if idx_theta.shape[-1] > n_elem > 1:
        idx_theta = idx_theta[..., :n_elem]

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


def do_softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh


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


class Embedding(torch.nn.Embedding):
    """Embedding layer wrapper that allows us to override the initialization scheme."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_method=None,
        **kwargs,
    ):
        self.init_method = init_method if init_method else None  # we'll use the default init
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.init_method is not None:
            self.init_method(self.weight)  # our init method
            self._fill_padding_idx_with_zero()  # the default implementation does this
        else:
            super().reset_parameters()  # the default implementation's init

    def forward(self, x, **kwargs):
        return super().forward(x)


# Temporary patch style, copied from old patch
class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()

        assert flash_attn_unpadded_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert rearrange is not None, "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

        # if self.training:
        # during training q,k,v always have same seqlen
        assert seqlen_k == seqlen_q

        is_causal = self.causal
        cu_seqlens_k = cu_seqlens_q
        dropout_p = self.dropout_p
        # else:
        #     # turn off FA causal mask after first inference autoregressive iteration
        #     # only on first autoregressive step q,k,v have same seqlen
        #     is_causal = seqlen_q == seqlen_k
        #     cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
        #                 device=q.device)
        #     dropout_p = 0

        output = flash_attn_unpadded_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )

        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output
