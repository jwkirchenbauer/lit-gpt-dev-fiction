import os
import json
import torch

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Union, Optional, Any, Literal

from litgpt.config import Config
from litgpt.config_dynamic import Config as DynamicConfig

from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class HuggingfaceConfig:
    """need to properly merge HF one day"""

    name: str
    checkpoint: Optional[str]
    block_size: Optional[int] = None
    strategy: Optional[str] = None

    @property
    def Block(self):
        if "llama" in self.name.lower():
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            return LlamaDecoderLayer
        else:
            raise ValueError("Provide the block name for this architecture.")

    def construct_model(self, objective, gradient_checkpointing: bool) -> torch.nn.Module:
        from axonn.models.transformers import parallelize

        source = self.checkpoint or self.name
        with parallelize(source) if self.strategy == "axonn_tp" else nullcontext():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(source))

        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        return model


@dataclass
class DataEntry:
    type: str
    prefix: str
    weight: int = 1.0
    data_signature: Optional[dict[str, list[str] | str]] = None
    name: Optional[str] = None
    data_dir: Optional[str] = None
    text_key: Optional[str] = None
    repetitions: Optional[int] = None
    max_epoch: Optional[int] = None
    scheduler: Optional[tuple[str, int]] = None
    return_data_id: Optional[bool] = None


@dataclass
class GoldfishConfig:
    strategy: Optional[str] = None  # off by default, set to "hash-table" or "hash-avalanche" to enable
    k: Union[None, int] = None  # GL k. Every k-th token will be dropped from the loss computation (`None` for no GL).
    start_position: int = 0  # GL start position. Start dropping tokens from this position.
    context_width: int = 13  # GL context width. Only for 'hash-table' strategy.


@dataclass
class FabricConfig:
    optimize_communication: Optional[bool] = False
    all_reduce_dtype: Optional[str] = None
    row_tensor_parallel_size: Optional[int] = 1
    col_tensor_parallel_size: Optional[int] = 1
    depth_tensor_parallel_size: Optional[int] = 1
    optim_sharding: Optional[bool] = False
    allow_optim_fusion: Optional[bool] = False
    use_apex_adamw: Optional[bool] = False


@dataclass
class CLISettings:
    # Main settings
    run_name: str = "default-run"  # The name for logging.
    out_dir: str = None  # type: ignore # The directory to save checkpoints. Required to be given or set as OUT_DIR
    resume: bool = True  # Whether to resume from a checkpoint in the out_dir.
    max_tokens: Optional[Union[int, float]] = None  # The maximum number of tokens to train on (determines max_steps).
    max_steps: Optional[int] = None  # Set max_tokens to zero if setting max_steps
    seed: int = 1337  # The random seed to use for reproducibility.

    # Model configuration
    model_name: str = "tiny-llama-1.1b"  # The model name to use when creating the model from config.py / config_dynamic
    model_impl: str = "litgpt"  # The model name to use when creating the model from config.py
    block_size: int = 2048  # The block size to use (lit-gpt-ese for sequence length).
    ignore_block_size_mismatch: bool = False  # Whether to ignore block size mismatch.
    model_checkpoint: Optional[str] = None  # The model checkpoint to load. Else, from config.
    doc_block_attn: bool = False  # Whether to mask out the attention between tokens from different documents.
    cache_attn: bool = False  # Whether to train the model with cache attention with cache tokens randomly inserted.
    eod_token: Optional[str] = None  # 'eos','bos','pad' The end-of-document token name (used for doc-block-attn).

    attn_impl: Literal["sdpa", "rocm"] = "rocm"  # The attention implementation to use.
    structured_init: bool = False  # Whether to use layer structured initialization for the model.
    structured_init_for_wte: bool = False  # Whether to use structured initialization for the input embedding layer.
    structured_init_olmo_variant: bool = False  # Whether to use olmo style structured initialization.

    # Training hyperparameters
    world_batch_size: int = 2048  # The total batch size across all devices and nodes.
    optimizer: str = "AdamW"
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=0.0004,  # The learning rate.
            weight_decay=0.1,  # The weight decay.
            betas=(0.9, 0.95),  # The beta parameters for the Adam optimizer.
            eps=1e-8,  # The eps parameter for the Adam optimizer
        )
    )
    grad_clip: float = 1.0  # The gradient clipping value.
    warmup_steps: int = 0  # The number of warmup steps.
    cooldown_steps: int = 0  # The number of cooldown steps.
    lr_schedule: str = "cosine"  # The learning rate schedule to use.
    min_lr: float = 0.00004  # The minimum learning rate to decay to.
    no_weight_decay_for_bias_and_norm_params: bool = False  # do not use weight decay for bias and norm params
    lr_scaler: Optional[str] = None  # The learning rate scaling strategy to use. "inverse_n_embd"

    # Objective and Regularization
    goldfish: GoldfishConfig = field(default_factory=lambda: GoldfishConfig())
    z_regularization: float = 0.0
    target_range_train: list[int] = (
        None  # the target range of ids to use when computing a cls loss using special tokens for the training data.
    )
    target_range_val: list[int] = None  # ...for the val data
    freeze_params: Optional[list[str]] = None  # List of parameter names to freeze (no gradients).
    use_jonas_ce: Optional[bool] = False  # Whether to use Jonas Kernel's custom cross-entropy loss.

    # Implementation and backend
    fabric_strategy: str = "ddp"  # The fabric strategy to use: ddp, fsdp, axonn_tp.
    fabric_precision: Literal["bf16-true", "bf16-mixed", "16-mixed", "16", "32"] = "bf16-mixed"
    fabric_use_lightning_environment: bool = False  # If False, use the auto setting, True, use LightningEnvironment.
    fabric: FabricConfig = field(
        default_factory=lambda: FabricConfig(
            **dict(
                optimize_communication=False,  # [Copilot] Whether to optimize communication.
                all_reduce_dtype=None,  # [Copilot] The dtype to use for all-reduce communication.
                row_tensor_parallel_size=1,  # The size of the row tensor parallel dimension
                col_tensor_parallel_size=1,  # The size of the col tensor parallel dimension
                depth_tensor_parallel_size=1,  # The size of the depth tensor parallel dimension
                optim_sharding=False,  # zero-1, activated directly in pytorch. May not play nicely with non-ddp
                allow_optim_fusion=False,  # fishes for fusion opportunities in the optimizer
            )
        )
    )
    micro_batch_size: int = 4  # The micro batch size to use.
    compile_model: bool = False  # Whether to compile the model.
    dynamo_ddp_config: Optional[Literal["ddp_optimizer", "python_reducer", "no_optimization"]] = None
    matmul_precision: str = "high"  # enable tf32 acc on cuda with this
    dataloader_num_workers: int = 0  # The number of workers to use for the dataloaders.
    n_chunks: int = 4  # The number of chunks to preload at a time from packed dataset.
    gradient_checkpointing: bool = False  # Whether to use activation checkpointing
    allow_nonfinite_loss: bool = False  # whether to end training immediately if non-finite loss is encountered
    use_liger_ce: bool = False  # Whether to use Liger Kernel's custom cross-entropy loss.

    # Logging
    logger_name: str = "wandb"  # The logger to use for logging, only supports "wandb" for now.
    wandb_offline: bool = True  # Whether to run wandb in offline mode (as we did on Frontier).
    logger_project: str = "tinyllama"  # The logger/wandb project to log to.
    wandb_tags: list[str] = field(default_factory=lambda: [])  # The tags to add the the wandb run.
    data_telemetry: bool = False  # Data telemetry switch, set based on needs.
    model_telemetry: bool = (
        False  # Whether to monitor important model values to look for spikes. May increase overhead. Induces compile warnings, ok/FIXME?
    )
    shape_watching_iters: int = 3  # Number of iterations to watch shapes for. Set to 0 to disable.
    log_step_interval: int = 1  # The base interval for logging (scales with gradient_accumulation_steps).
    eval_iters: int = 100  # The number of iterations to process during a validation loop.
    save_step_interval: int = 2000  # The number of iterations between saving.
    eval_step_interval: int = 2000  # The number of iterations between evaluating.
    save_first_step: bool = False  # Whether to save the checkpoint at the first step
    save_last_step: bool = False  # Whether to save the checkpoint at the last step
    save_n_min_before_job_done: Optional[int] = None  # Save the checkpoint n minutes before current job done
    sanity_validate: bool = False  # Whether to run a short sanity check validation loop at the start.
    measure_utilization: bool = False  # Print FLOPs and MFU. Flaky on Frontier, so defaulting to False, FIXME?
    estimate_param_count: bool = False  # Estimate the number of parameters in the model using a function.
    simple_gptneox_tflops: bool = False  # Use a simple GPT-NeoX flops calculation. Standin on Frontier.
    peak_tflops_per_device: float = (
        192.0  # The peak TFLOPS per device for the GPUS on this system. default is Frontier's  MI250X
    )
    derive_cost_basis: bool = False  # Derive the cost basis for run on this topology.
    cards_per_node: int = 8  # The number of cards per node.
    validate_only: bool = False  # Whether to only run validation.
    initial_validate: bool = False  # Whether to run a validation loop when trainig starts.
    log_scaling_law_metrics: bool = False  # Whether to log the width and depth etc.
    stability_step: Optional[int] = None  # The step at which we log "stable run"

    # Data Handling
    # PKDS arguments:
    shuffle_filenames: bool = True  # (PKDS only.) Shuffle filenames glob'd up for each prefix
    shuffle_blocks: bool = True  # (PKDS only.) Whether to shuffle the blocks in files.
    all_block_size_tensors: bool = False  # Assume all datasets return tensors of exactly block_size
    # HFDS arguments:
    pad_to_block_size: bool = False  # Whether to pad to the block size (HFDS only).
    add_bos: bool = True  # Whether to add the BOS token to the input (HFDS only).
    add_eos: bool = True  # Whether to add the EOS token to the input (HFDS only).
    data_signature: dict[str, list[str] | str] = field(
        default_factory=lambda: {"keys": ["text"], "format_fn": "pass_text"}
    )  # The data signature to use for processing rows of the dataset. Can be set individually per dataset. (HFDS only).
    # For both backends:
    collate_checks_enabled: bool = True  # Enable checks for the collate function.
    all_block_size_tensors: bool = False  # Assume all datasets return tensors with the same size, may reduce latency.
    use_chat_template: bool = False  # Whether to use the chat template in the collator.
    return_data_id: bool = False  # Whether to return the data_id in the dataset.
    data_config: Union[str, dict[str, list[DataEntry]]] = field(
        default_factory=lambda: {
            "train_data": [DataEntry("pkds", "", 1)],
            "val_data": [DataEntry("pkds", "", 1)],
        }
    )
    # The directories containing the training/validation data.
    train_data_dir: str = "$DATA_DIR/spj_star_combined_full_tinyllama_tokd"
    val_data_dir: str = "$DATA_DIR/spj_star_combined_full_tinyllama_tokd"
    # The path to the tokenizer to use [required to identify pad_token_id even for pkds]
    tokenizer_path: str = (
        "/lustre/orion/csc569/proj-shared/language_models/external/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )
    # For exact match memorization validation logic
    memorization_validation: bool = False
    prefix_lengths: Union[dict[str, int], list[int]] = field(
        default_factory=lambda: {"min": 50, "max": 150, "step": 50}
    )
    suffix_lengths: Union[dict[str, int], list[int]] = field(
        default_factory=lambda: {"min": 25, "max": 75, "step": 25},
    )

    model_config: Union[Config, DynamicConfig, HuggingfaceConfig] = field(init=False)
    model_overwrite: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate arguments
        if self.out_dir is None:
            self.out_dir = os.getenv("OUTPUT_DIR", "NOT_FOUND")
        assert self.out_dir != "NOT_FOUND"
        assert self.tokenizer_path, "Tokenizer has to be specified."

        # Handle data config
        self._parse_data_config()
        self._process_data_entries()
        self._expand_paths()

        # Handle memorization validation
        self._complete_memorization_validation()

        # Handle fabric config
        self._complete_fabric_config()
        # Tensor parallelism is implemented by the AxoNN fabric only.
        if (
            self.fabric.depth_tensor_parallel_size > 1
            or self.fabric.row_tensor_parallel_size > 1
            or self.fabric.col_tensor_parallel_size > 1
        ):
            assert self.fabric_strategy == "axonn_tp", "x_tensor_parallel_size > 1 implies use of axonn_tp."

        self._parse_environment_variables()

        # Add any derived cfg here
        self.node_batch_size = self.world_batch_size // self.num_nodes
        self.loader_block_size = self.block_size + 1
        self.global_total_time = 0
        self.max_tokens_per_device = 0
        self.tokens_per_step = 0

        self.batch_size = self.node_batch_size // self.devices
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        self.warmup_iters = self.warmup_steps * self.gradient_accumulation_steps
        self.cooldown_iters = self.cooldown_steps * self.gradient_accumulation_steps

        self.log_iter_interval = self.log_step_interval * self.gradient_accumulation_steps
        self.dataset_names = [i.prefix for i in self.data_config["train_data"]]

        self._validate_args()

        # Finally, store model config object itself
        if self.model_impl == "litgpt":
            self.model_config = Config.from_name(self.model_name, **self.model_overwrite)
        elif self.model_impl == "dynamic":
            self.model_config = DynamicConfig.from_name(self.model_name, **self.model_overwrite)
        elif self.model_impl == "huggingface":
            self.model_config = HuggingfaceConfig(self.model_name, **self.model_overwrite)
            self.model_config.block_size = self.block_size

        # Set strategy
        self.model_config.strategy = self.fabric_strategy
        # Set attn_impl
        self.model_config.attn_impl = self.attn_impl

        # check whether we're requesting a compatible modeling config and attn_impl
        if self.model_config.surrogate_config:
            assert self.attn_impl == "sdpa", "Surrogate models only support SDPA attention."

        # Set structured_init
        self.model_config.structured_init = self.structured_init
        self.model_config.structured_init_for_wte = self.structured_init_for_wte
        self.model_config.structured_init_olmo_variant = self.structured_init_olmo_variant

    def _validate_args(self):

        if self.max_tokens is not None:
            assert self.max_tokens % 1 == 0, "max_tokens must be an integer"
            self.max_tokens = int(self.max_tokens)

        assert ((self.max_steps is not None) and (self.max_steps > 0)) ^ (
            ((self.max_tokens is not None) and (self.max_tokens > 0))
        ), f"only max_steps ({self.max_steps}) xor max_tokens ({self.max_tokens}) can be specified"
        assert len(set(self.dataset_names)) == len(
            self.data_config["train_data"]
        ), "please provide different names for each subset"

        # Any additional sanity checks here.
        assert self.gradient_accumulation_steps > 0, "derived gradient_accumulation_steps must be > 0"
        assert (
            self.world_batch_size
            == self.micro_batch_size * self.gradient_accumulation_steps * self.devices * self.num_nodes
        ), "world batch size should be: micro_batch_size * gradient_accumulation_steps * devices * num_nodes"

        assert not ((self.goldfish.strategy is None) ^ (self.goldfish.k is None)), "both GL param must be set or None"

        assert not (
            self.memorization_validation and self.target_range_val
        ), "both memorization_validation and target_range_val cannot be set at same time"

        if self.fabric_strategy == "ddp" and self.compile_model and self.gradient_checkpointing:
            assert (
                self.dynamo_ddp_config == "python_reducer"
            ), "dynamo_ddp_config must be python_reducer for this setup."
            # NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph. Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.

    def _parse_environment_variables(self):
        """Parse env variables and directly store as non-field attributes"""
        self.SLURM_JOB_ID = int(os.getenv("SLURM_JOB_ID", 0))
        self.SLURM_ARRAY_JOB_ID = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.SLURM_ARRAY_TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        self.SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))
        self.MASTER_ADDR = os.getenv("MASTER_ADDR", "0")
        self.MASTER_PORT = int(os.getenv("MASTER_PORT", 0))
        self.WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        self.RANK = int(os.getenv("SLURM_PROCID", "0"))
        self.devices = int(os.getenv("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
        self.num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))

    def _parse_data_config(self) -> dict[str, list[DataEntry]]:
        """If data_config is a string, load it from a file."""
        if isinstance(self.data_config, str):
            try:
                with open(self.data_config, mode="r") as json_file:
                    self.data_config = json.load(json_file)
            except Exception as e:
                raise ValueError(
                    f"data_config passed was a string, but failed to load as a json object from {self.data_config}: {e}"
                )

    def _process_data_entries(self):
        """If they are dicts, convert them to DataEntry objects."""
        processed_data_config = {"train_data": [], "val_data": []}
        unpack_entry = lambda entry: DataEntry(**entry) if isinstance(entry, dict) else entry
        for entry in self.data_config["train_data"]:
            processed_data_config["train_data"].append(unpack_entry(entry))
        for entry in self.data_config["val_data"]:
            processed_data_config["val_data"].append(unpack_entry(entry))
        self.data_config = processed_data_config

    def _expand_paths(self):
        """Materialize fully qualified paths."""
        self.train_data_dir = os.path.expandvars(self.train_data_dir) if self.train_data_dir is not None else ""
        self.val_data_dir = os.path.expandvars(self.val_data_dir) if self.val_data_dir is not None else ""
        for entry in self.data_config["train_data"] + self.data_config["val_data"]:
            if entry.data_dir is not None:
                entry.data_dir = os.path.expandvars(entry.data_dir)

    def _complete_fabric_config(self):
        """Complete fabric config with missing values if only partially specified."""
        self.fabric = FabricConfig(**self.fabric) if isinstance(self.fabric, dict) else self.fabric

    def _complete_memorization_validation(self):
        if isinstance(self.prefix_lengths, dict):
            min_prefix_len, max_prefix_len, step = (
                self.prefix_lengths["min"],
                self.prefix_lengths["max"],
                self.prefix_lengths.get("step", 1),
            )
            prefix_lengths = list(range(min_prefix_len, max_prefix_len + 1, step))
        elif isinstance(self.prefix_lengths, list):
            prefix_lengths = sorted(self.prefix_lengths)
        else:
            raise ValueError(f"prefix_lengths must be a dict or list, got {self.prefix_lengths}")

        if isinstance(self.suffix_lengths, dict):
            min_suffix_len, max_suffix_len, step = (
                self.suffix_lengths["min"],
                self.suffix_lengths["max"],
                self.suffix_lengths.get("step", 1),
            )
            suffix_lengths = list(range(min_suffix_len, max_suffix_len + 1, step))
        elif isinstance(self.suffix_lengths, list):
            suffix_lengths = sorted(self.suffix_lengths)
        else:
            raise ValueError(f"suffix_lengths must be a dict or list, got {self.suffix_lengths}")

        self.prefix_lengths = prefix_lengths
        self.suffix_lengths = suffix_lengths

        if self.memorization_validation:
            print(f"Coercing return_data_id globally to True for memorization validation.")
            self.return_data_id = True


@dataclass
class CLISettingsDatasetOptimizer(CLISettings):
    alpha_iter_interval: int = 0
    theta_tn_setting: str = (
        "tstar"  # The init setting for the theta step within alpha stage, either "t0", "tn", or "tstar".
    )
    restore_model_optim_after_alpha_stage: bool = True
    finalize_at_t_star: bool = False
    t_alpha: int = 10

    rank_wise_data_mix: bool = True  # Whether to mix the data rank wise or not.
    rank_wise_data_priors: bool = True  # Whether to use rank wise data priors or not.
    default_alpha: float = 1.0  # The default alpha value to use.

    alpha_optimizer: str = "AdamW"
    alpha_optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=0.0004,  # The learning rate.
            weight_decay=0.1,  # The weight decay.
            betas=(0.9, 0.95),  # The beta parameters for the Adam optimizer.
            eps=1e-8,  # The eps parameter for the Adam optimizer
        )
    )
    default_norm_type: float = 2.0  # The default norm type to use when measuring grads.
    alpha_grad_clip: float = 1e9  # The gradient clipping value, which for alpha is to default as a no-op.
    alpha_warmup_steps: int = 0  # The number of warmup steps.
    alpha_lr_schedule: str = "cosine"  # The learning rate schedule to use.
    alpha_min_lr: float = 0.00004  # The minimum learning rate to decay to.
    backtracking: bool = False  # Whether to use backtracking line search for the alpha optimization lr.

    alpha_log_grad_bucket_norm_steps: int = None  # freq to log the norm of the gradient buckets.

    alpha_accum_grad_lr_floor: float = 1e-12  # The small floor value to use if accum grads * lr.
    alpha_accum_grads_w_lr: bool = False  # Whether to accumulate gradients multiplied by the learning rate for alpha.
    alpha_accum_average_grad: bool = False  # Whether to accumulate the average of gradients for use in alpha stage.

    alpha_log_step_interval: int = 1  # The base interval for alpha logging.

    alpha_constraint: list[Union[float, None]] = field(
        default_factory=lambda: [None, None]
    )  # The constraint pair (min, max) to use for all alphas.

    alpha_renormalization_strategy: str = None  # The strategy to use for renormalizing the alphas.

    export_alpha_mixture: bool = (
        False  # Whether to load the most recent checkpoint, copy data config, and export alpha mixture as its rank_wise_data_priors.
    )

    def __post_init__(self):
        super().__post_init__()
        self.alpha_warmup_iters = self.alpha_warmup_steps  # no gradient accumulation for alpha
        self.alpha_log_iter_interval = self.alpha_log_step_interval  # no gradient accumulation for alpha
        self.alpha_log_grad_bucket_norm_iters = self.alpha_log_grad_bucket_norm_steps
        assert self.alpha_iter_interval % self.gradient_accumulation_steps == 0


@dataclass
class DataEntryDatasetOptimizer(DataEntry):
    initial_alpha: Optional[float] = None
