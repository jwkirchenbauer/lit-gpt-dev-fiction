"""
This script is originally adapted from and inspired by the tinyllama.py and
redpajama.py scripts in the lit-gpt/pretrain directory.

The lit-gpt authors designed this such that setup -> train reads ~linearly.
"""

####################################################################################################
# Imports.
####################################################################################################

import time

global_start_time = time.time()
import math
import os
import socket
import gc

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Optional
import json
import random

import torch
import torch.nn as nn

DEVICE_NAME = None
DEVICE_DRIVER_VERSION = None
try:
    DEVICE_NAME = torch.cuda.get_device_name()  # shouldn't fail even on AMD, except maybe old torch
    DEVICE_DRIVER_VERSION = torch.version.cuda if torch.version.cuda else torch.version.hip
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Device found: {DEVICE_NAME}, running version {DEVICE_DRIVER_VERSION}.")
except RuntimeError as e:
    assert "no NVIDIA driver" in str(e), "On AMD, for old torch, device inquiry may fail, but not other errors."

if TYPE_CHECKING:
    import torch.distributed
    import torch.version
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from litgpt.settings import CLISettings


from litgpt.tokenizer import Tokenizer
from litgpt.packed_cycle_dataset import CombinedDataset, PackedDataset
from litgpt.huggingface_dataset import HuggingfaceDataset
from litgpt.data_loading_utils import generic_collate_fn
import litgpt.utils
from litgpt.utils import simple_gptneox_tflops, param_count_estimator
from litgpt.data_scheduler_utils import DataSchedulerTracker, DataScheduler
from litgpt.doc_block_utils import get_ltor_masks_and_position_ids, get_cache_attn_masks
from litgpt.monitor import (
    enable_monitoring_on_step,
    disable_monitoring_and_retrieve_metrics,
    track_gradient_metrics,
    get_MFU_metrics,
)


def scale_lr(lr, n_embd=None, n_layer=None, lr_scaler=None):
    if lr_scaler is None:
        pass
    elif lr_scaler == "inverse_n_embd":
        lr = lr / n_embd
    elif lr_scaler == "inverse_sqrt_n_embd":
        lr = lr / math.sqrt(n_embd)
    elif lr_scaler == "inverse_n_embd_sqrt_layer":
        lr = lr / (n_embd * math.sqrt(n_layer))
    else:
        raise ValueError(f"Unsupported lr_scaler: {lr_scaler}")
    return lr


def get_param_groups(model, no_weight_decay_for_bias_and_norm_params=True, no_wd_on_embedding=False, cfg=None):
    # takes model instead of model.named_parameters() to allow for addressing different parts of the model
    # calls litgpt.optim.get_param_groups(model_part.named_parameters(), cfg.no_weight_decay_for_bias_and_norm_params)
    # then returns the concatenated list of param_groups

    # in particular, we just want to insert a "scaler" into the param_groups for each of these parts
    # and we'll use that at runtime to adjust the learning rate for each group
    assert cfg is not None, "cfg must be passed to get_param_groups"

    wte_param_groups = litgpt.optim.get_param_groups(
        model.transformer.wte.named_parameters(), no_weight_decay_for_bias_and_norm_params, no_wd_on_embedding
    )
    h_param_groups = litgpt.optim.get_param_groups(
        model.transformer.h.named_parameters(), no_weight_decay_for_bias_and_norm_params, no_wd_on_embedding
    )
    ln_f_param_groups = litgpt.optim.get_param_groups(
        model.transformer.ln_f.named_parameters(), no_weight_decay_for_bias_and_norm_params, no_wd_on_embedding
    )
    lm_head_param_groups = litgpt.optim.get_param_groups(
        model.lm_head.named_parameters(), no_weight_decay_for_bias_and_norm_params, no_wd_on_embedding
    )

    scaler_partial = partial(
        scale_lr, n_embd=cfg.model_config.n_embd, n_layer=cfg.model_config.n_layer, lr_scaler=cfg.lr_scaler
    )

    for group in wte_param_groups:
        group["scaler"] = scaler_partial
        group["group_name"] = "wte"
    for group in h_param_groups:
        group["scaler"] = scaler_partial
        group["group_name"] = "h"
    for group in ln_f_param_groups:
        group["scaler"] = scaler_partial
        group["group_name"] = "ln_f"
    for group in lm_head_param_groups:
        group["scaler"] = scaler_partial
        group["group_name"] = "lm_head"

    all_param_groups = wte_param_groups + h_param_groups + ln_f_param_groups + lm_head_param_groups
    return all_param_groups


from dataclasses import asdict, is_dataclass
from jsonargparse import CLI
import re


end_time = time.time()
if int(os.getenv("SLURM_PROCID", "0")) == 0:
    print(f"Time to load libraries: {end_time - global_start_time:.02f} seconds.")

####################################################################################################
# Setup functions.
####################################################################################################
Fabric = litgpt.utils.LightningFabric | litgpt.utils.SimpleFabric


def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    # Do they AMD cards pick up on any of this? :
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    # Dynamo + DDP primitives:
    if cfg.dynamo_ddp_config is not None:
        torch._dynamo.config.optimize_ddp = cfg.dynamo_ddp_config


def setup_fabric(cfg: CLISettings) -> Fabric:
    """Sets up the fabric and logger based on the cfg."""
    # Instantiate the logger.
    if cfg.logger_name == "wandb":
        # set offline dynamically from environemtn
        logger = WandbLogger(
            project=cfg.logger_project,
            name=cfg.run_name,
            save_dir=cfg.out_dir,
            tags=cfg.wandb_tags,
            offline=cfg.wandb_offline,
        )
    else:
        raise ValueError(f"`logger={cfg.logger_name}` is not a valid option.")

    # Instantiate the fabric.
    if cfg.fabric_strategy == "ddp-simple":
        assert cfg.num_nodes == 1
        fabric = litgpt.utils.SimpleFabric(precision=cfg.fabric_precision, loggers=[logger])
        fabric.print(f"Using SimpleFabric with strategy {cfg.fabric_strategy}")
    else:
        if "fsdp" in cfg.fabric_strategy:
            sharding_strategy = (
                "SHARD_GRAD_OP"  # SHARD_GRAD_OP can be nice on small machines
                if "grad" in cfg.fabric_strategy  # USE "HYBRID_SHARD" AT SCALE  # choose FULL_SHARD if oom
                else "FULL_SHARD" if "full" in cfg.fabric_strategy else "HYBRID_SHARD"
            )

            sharding_strategy = "HYBRID_SHARD"
            from torch.distributed.device_mesh import init_device_mesh

            mesh_2d = init_device_mesh("cuda", (cfg.num_nodes, cfg.devices))

            strategy = FSDPStrategy(
                auto_wrap_policy={cfg.model_config.Block},
                mixed_precision=derive_precision(cfg.fabric_precision, cfg.fabric),
                activation_checkpointing_policy={cfg.model_config.Block} if cfg.gradient_checkpointing else None,
                state_dict_type="full",
                sharding_strategy=sharding_strategy,
                device_mesh=mesh_2d,
                param_init_fn=((lambda x: x.to_empty(recurse=False)) if cfg.model_impl == "huggingface" else None),
            )
        elif cfg.fabric_strategy == "ddp":
            strategy = DDPStrategy()
        elif cfg.fabric_strategy == "single":
            strategy = SingleDeviceStrategy(
                device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            )
        elif cfg.fabric_strategy == "axonn_tp":
            from axonn.lightning import AxonnStrategy
            from axonn import axonn as ax

            def global_rank_for_creating_dataloader(self):
                return ax.config.G_intra_d * ax.config.data_parallel_rank + ax.config.intra_layer_depth_parallel_rank

            def global_world_size_for_creating_dataloader(self):
                return ax.config.G_intra_d * ax.config.G_data

            AxonnStrategy.global_rank_for_creating_dataloader = global_rank_for_creating_dataloader
            AxonnStrategy.global_world_size_for_creating_dataloader = global_world_size_for_creating_dataloader

            strategy = AxonnStrategy(
                G_intra_r=cfg.fabric.row_tensor_parallel_size,
                # G_intra_c=cfg.fabric.col_tensor_parallel_size,
                G_intra_d=cfg.fabric.depth_tensor_parallel_size,
                overlap_communication=cfg.fabric.optimize_communication,
            )
        else:
            raise ValueError(f"`fabric_strategy={cfg.fabric_strategy}` is not a valid option.")

        # Instantiate and launch/initialize the fabric distributed environment management.
        fabric = litgpt.utils.LightningFabric(
            devices=cfg.devices,
            strategy=strategy,
            precision=cfg.fabric_precision,
            loggers=[logger],
            num_nodes=cfg.num_nodes,
        )
        fabric.print(f"Using LightningFabric with strategy {cfg.fabric_strategy} ")
        fabric.launch()

    fabric.print(f"> gradient_accumulation_steps = {cfg.gradient_accumulation_steps}")
    fabric.print(f"> micro_batch_size = {cfg.micro_batch_size}")
    fabric.print(f"> global_batch_size = {cfg.world_batch_size}")

    if cfg.fabric_strategy == "axonn_tp":
        from axonn import axonn as ax

        fabric.print(f"> cfg.world_batch_size = {cfg.world_batch_size}")
        fabric.print(
            f"> micro_bs*depth*data = {cfg.micro_batch_size} * {ax.config.G_intra_d} * {ax.config.G_data} = {cfg.micro_batch_size * ax.config.G_intra_d * ax.config.G_data}"
        )
        fabric.print(
            f"> micro_bs*world_size/row = {cfg.micro_batch_size} * {fabric.world_size} / {ax.config.G_intra_r} ={cfg.micro_batch_size * fabric.world_size / ax.config.G_intra_r}"
        )
        assert (cfg.micro_batch_size * ax.config.G_intra_d * ax.config.G_data) == (
            cfg.micro_batch_size * fabric.world_size / ax.config.G_intra_r
        ), "math aint mathing"

    return fabric


####################################################################################################
# Main driver functions.
####################################################################################################


def startup(fabric: Fabric, cfg: CLISettings):
    """The main driver function for the training script."""
    start_time = time.time()

    # Get job remaining time
    if cfg.save_n_min_before_job_done is not None:
        if fabric.global_rank == 0:
            global_total_time = _get_time_from_slurm()
            fabric.print(f"Total job time: {global_total_time:.02f} seconds.")
        else:
            global_total_time = 0

        global_total_time = fabric.broadcast(global_total_time, 0)  # does this have to be a broadcast?
        cfg.global_total_time = global_total_time

    # Prepare directories for logging
    if fabric.global_rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.out_dir) / fabric.get_prefix_for_checkpoint()).mkdir(parents=True, exist_ok=True)
        # Last step before we move on is to dump the cfg to a file in the out_dir.
        # This is is itself loadable as a config by passing like train.py --config run_config.json
        with open(f"{cfg.out_dir}/run_config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=4)
        with open(f"{cfg.out_dir}/model_config.json", "w") as f:
            json.dump(asdict(cfg.model_config) if is_dataclass(cfg.model_config) else cfg.model_config, f, indent=4)
    # Load tokenizer
    tokenizer = Tokenizer(cfg.tokenizer_path)
    if tokenizer.pad_id is None:
        tokenizer.pad_id = -1
    if cfg.cache_attn:
        assert tokenizer.cache_token_id is not None
    if cfg.doc_block_attn:
        assert tokenizer.eod_token_id is not None

    # Create data objects
    t0 = time.time()
    # On block size, moved this here to be more explicit that this is happening ...
    if not cfg.ignore_block_size_mismatch:
        assert cfg.block_size == cfg.model_config.block_size, "cfg.block_size must match config.block_size"
    # Increase by one to actually be supervising "block_size" tokens in every update after rshift.
    train_dataloader, val_dataloader, data_scheduler_tracker, val_data_scheduler_tracker = create_dataloaders(
        batch_size=cfg.micro_batch_size,
        block_size=cfg.loader_block_size,
        fabric=fabric,
        seed=(cfg.seed + fabric.global_rank),
        cfg=cfg,
        tokenizer=tokenizer,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    data_scheduler = DataScheduler(data_scheduler_tracker, cfg.data_config["train_data"], cfg)
    val_data_scheduler = DataScheduler(val_data_scheduler_tracker, cfg.data_config["val_data"], cfg)
    data_scheduler.step(0)
    val_data_scheduler.step(0)  # we will also reset this each time we validate
    fabric.print(f"Time to instantiate and setup dataloaders: {time.time() - t0:.02f} seconds.")

    # Construct the model
    fabric.seed_everything(cfg.seed)  # same seed for every process to init model (FSDP)
    if cfg.model_checkpoint is not None:
        litgpt.utils.check_valid_checkpoint_dir(Path(cfg.model_checkpoint))
    fabric.print(f"Loading model with {cfg.model_config.__dict__}")

    # Set the objective
    objective = dict(
        op=litgpt.utils.chunked_cross_entropy,
        ignore_index=tokenizer.pad_id,
        z_regularization=cfg.z_regularization,
        gl_k=cfg.goldfish.k,
        gl_strategy=cfg.goldfish.strategy,
        gl_start_position=cfg.goldfish.start_position,
        gl_context_width=cfg.goldfish.context_width,
        target_range=cfg.target_range_train,
        return_logits_targets=False,
        use_jonas_ce=cfg.use_jonas_ce,
        use_liger_ce=cfg.use_liger_ce,
    )

    # Initialize the model
    t0 = time.time()
    with fabric.init_module(empty_init=cfg.fabric_strategy == "fsdp"):
        model = cfg.model_config.construct_model(
            objective=objective, gradient_checkpointing=cfg.gradient_checkpointing and cfg.fabric_strategy != "fsdp"
        )
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.")

    # before (optional compile) distribution and setup
    if cfg.freeze_params:
        fabric.print("Freezing up the following parameters:")
        for name, param in model.named_parameters():
            if name in cfg.freeze_params:
                param.requires_grad = False
                fabric.print(f"{name}.requires_grad? {param.requires_grad}")
    else:
        fabric.print("No parameters frozen.")

    num_params = litgpt.utils.num_parameters(model)

    fabric.log_to_summary({"num_parameters": num_params, "device": DEVICE_NAME})

    # With fabric and the model up, we can compute a few last derived cfg
    if cfg.max_steps is None:
        # because we don't really trust our token counting under row col parallelism currently
        assert cfg.fabric.row_tensor_parallel_size == 1, "max_steps must be set if row_tensor_parallel_size > 1"
        assert cfg.fabric.col_tensor_parallel_size == 1, "max_steps must be set if col_tensor_parallel_size > 1"

        cfg.max_tokens_per_device = cfg.max_tokens // fabric.world_size
        cfg.tokens_per_step = cfg.micro_batch_size * cfg.block_size
        cfg.max_steps = cfg.max_tokens_per_device // cfg.tokens_per_step

    if cfg.compile_model:
        model = torch.compile(model)

    # Set up the final fabric+model details
    t0 = time.time()
    model = fabric.setup(model)
    fabric.print(f"Model with full setup is:\n{model}")
    fabric.print(f"Total parameters: {num_params:,}")

    if cfg.estimate_param_count or cfg.log_scaling_law_metrics:
        param_count_estimate = param_count_estimator(
            width=cfg.model_config.n_embd,
            depth=cfg.model_config.n_layer,
            vocab_size=cfg.model_config.padded_vocab_size,
            n_head=cfg.model_config.n_head,
            head_size=cfg.model_config.head_size,
            n_query_groups=cfg.model_config.n_query_groups,
            intermediate_size=cfg.model_config.intermediate_size,
        )
        cfg.param_count_estimate = param_count_estimate
        fabric.print(
            f"Model parameter count check: napkin math == trainable params ? {param_count_estimate == num_params}\n {param_count_estimate:,} == {num_params:,} "
        )
        if cfg.fabric_strategy != "axonn_tp":  # it doesnt work when the model is sharded
            assert param_count_estimate == num_params, "Model parameter count check failed."

        if cfg.log_scaling_law_metrics:
            scaler_partial = partial(
                scale_lr, n_embd=cfg.model_config.n_embd, n_layer=cfg.model_config.n_layer, lr_scaler=cfg.lr_scaler
            )
            scaling_laws_measures = {
                "scaling/params": cfg.param_count_estimate,
                "scaling/width": cfg.model_config.n_embd,
                "scaling/depth": cfg.model_config.n_layer,
                "scaling/width_depth_ratio": cfg.model_config.n_embd / cfg.model_config.n_layer,
                "scaling/max_lr": cfg.optim_config["lr"],
                "scaling/min_lr": cfg.min_lr,
                "scaling/max_lr_scaled": scaler_partial(cfg.optim_config["lr"]),
                "scaling/min_lr_scaled": scaler_partial(cfg.min_lr),
                "scaling/model_id": f"{cfg.model_config.n_embd}x{cfg.model_config.n_layer}",
                "scaling/mup_strat_id": f"{cfg.model_config.n_embd}x{cfg.model_config.n_layer}_{cfg.lr_scaler}",
            }
            fabric.log_to_summary(scaling_laws_measures)

    fabric.print(f"Time to setup model: {time.time() - t0:.02f} seconds.")

    t0 = time.time()
    # Set up the optimizer and training state object.
    # param_groups = litgpt.optim.get_param_groups(model.named_parameters(), cfg.no_weight_decay_for_bias_and_norm_params)
    param_groups = get_param_groups(model, cfg.no_weight_decay_for_bias_and_norm_params, cfg=cfg)
    # NOTE: Fusion caused slowdowns during GB with massive models on larger topologies.
    # NOTE: optim_sharding + axonn does not play nicely (loss doesnt go down) FIXME
    optimizer = litgpt.optim.get_optimizer(
        cfg.optimizer,
        model=model,
        pytorch_optimizer_sharding=cfg.fabric.optim_sharding,
        allow_fusion=(cfg.fabric.allow_optim_fusion and "bf16" in cfg.fabric_precision),
        use_apex_adamw=cfg.fabric.use_apex_adamw,
    )(param_groups, **cfg.optim_config)
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.print(f"Time to instantiate and setup optimizers: {time.time() - t0:.02f} seconds.")

    state = {
        "model": model,
        "optimizer": optimizer,
        "tokenizer": tokenizer,
        "data_scheduler": data_scheduler,
        "val_data_scheduler": val_data_scheduler,
        "microbatch_step": 0,  # mbs steps
        "optimizer_step": 0,  # optimizer updates taken
    }

    t0 = time.time()
    # If resuming, determine the checkpoint to resume from.
    resume_ckpt = load_checkpoint(
        fabric, state, cfg.out_dir, cfg.run_name, cfg.model_checkpoint, cfg.model_impl, cfg.resume, cfg.fabric_strategy
    )
    fabric.print(f"Time to load model checkpoint: {time.time() - t0:.02f} seconds.")

    # Report the full cfg set for the run.
    fabric.print(f"cmdline + derived cfg:\n{json.dumps(cfg.__dict__, default=lambda x:x.__dict__, indent=4)}")
    fabric.logger.log_hyperparams(cfg.__dict__)

    end_time = time.time()
    fabric.print(f"Total time to run main func setups: {end_time - start_time:.02f} seconds.")

    return state, train_dataloader, val_dataloader, data_scheduler, val_data_scheduler, resume_ckpt


@torch.no_grad()
def validate(
    fabric: Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    val_data_scheduler: DataScheduler,
    max_validation_steps: int,
    tokenizer: Tokenizer,
    cfg,
    train_step: int = None,
) -> torch.Tensor:
    if val_dataloader is None:
        return torch.as_tensor(float("-Inf"))
    fabric.print(f"Validating for {max_validation_steps} steps ...")
    model.eval()

    if cfg.target_range_val is not None:
        orig_objective = model.objective
        val_objective = dict(
            op=litgpt.utils.chunked_cross_entropy,
            ignore_index=tokenizer.pad_id,
            gl_k=None,  # cfg.goldfish.k, we don't want to use goldfish for validation
            gl_strategy=None,  # cfg.goldfish.strategy,
            gl_start_position=None,  # cfg.goldfish.start_position,
            gl_context_width=None,  # cfg.goldfish.context_width,
            target_range=cfg.target_range_val,
            return_logits_targets=True,  # we use this to compute prediction accuracy when its a classification task
        )
        model.set_objective(val_objective)
        correct_predictions = torch.zeros(max_validation_steps * cfg.micro_batch_size, device=fabric.device)
    else:
        orig_objective = None
        val_objective = None
        correct_predictions = None

    em_stats_list = []
    losses = torch.full((max_validation_steps,), -1.0, device=fabric.device)
    sample_wise_losses = torch.full((max_validation_steps * cfg.micro_batch_size,), -1.0, device=fabric.device)
    metadata_list = torch.full((max_validation_steps * cfg.micro_batch_size,), -1, device=fabric.device)
    ds_meta_to_int = {ds_metadata: i for i, ds_metadata in enumerate(cfg.val_dataset_prefixes)}
    ds_meta_from_int = {i: ds_metadata for i, ds_metadata in enumerate(cfg.val_dataset_prefixes)}

    # reset the val data scheduler and tracker
    val_data_scheduler.step(0)
    val_data_scheduler.data_scheduler_tracker.reset()

    val_iterator = iter(val_dataloader)
    total_steps_taken = max_validation_steps
    for k in range(max_validation_steps):
        # this position mimics the training scheduler's post-step call.
        # it starts at 0, and we want to use the schedule according to the value at k
        val_data_scheduler.step(k)

        try:
            data_batch = next(val_iterator)
        except StopIteration:
            # If this is caught, the val data config must be such that we've exhausted it
            # in fewer than max_validation_steps.
            print(
                f"Validation data exhausted before step {k}/{max_validation_steps} on rank ({fabric.global_rank}/{fabric.world_size})"
            )
            total_steps_taken = k  # this is last valid k + 1, count of steps taken
            break
        # input_ids, labels = data
        input_ids, labels, metadata = data_batch
        input_ids = input_ids.to(fabric.device, non_blocking=True)
        labels = labels.to(fabric.device, non_blocking=True)

        mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)
        outputs = model(
            input_ids,
            position_ids=positions,
            labels=labels,
            attention_mask=mask,
            # return_logits=(cfg.memorization_validation),
            return_logits=True,
            reduction=None,
        )
        if correct_predictions is None:
            losses[k] = outputs["loss"].mean()
        else:
            # we're getting back logits and targets as well
            loss, logits, targets = outputs["loss"].mean()
            losses[k] = loss
            # compute the accuracy
            correct_predictions[k * cfg.micro_batch_size : (k + 1) * cfg.micro_batch_size] = (
                torch.argmax(logits, dim=-1) == targets
            )

        if cfg.memorization_validation:
            batch_em_stats = litgpt.utils.batch_exact_match(
                outputs["logits"], labels, metadata, cfg, tokenizer, step_i=train_step
            )
        else:
            batch_em_stats = {}
        em_stats_list.append(batch_em_stats)

        # calculate sample_wise_losses
        # curr_logits = outputs["logits"].reshape(-1, outputs["logits"].shape[-1])
        # curr_labels = labels
        # token_wise_loss = nn.functional.cross_entropy(
        #     curr_logits, curr_labels.reshape(-1), ignore_index=tokenizer.pad_id, reduction="none"
        # )
        # token_wise_loss = token_wise_loss.reshape(labels.shape[0], -1)
        # valid_mask = (curr_labels != tokenizer.pad_id).float()
        # sample_loss = (token_wise_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        # sample_wise_losses[k * cfg.micro_batch_size : (k + 1) * cfg.micro_batch_size] = sample_loss
        sample_wise_losses[k * cfg.micro_batch_size : (k + 1) * cfg.micro_batch_size] = outputs["loss"]
        metadata_list[k * cfg.micro_batch_size : (k + 1) * cfg.micro_batch_size] = torch.tensor(
            [ds_meta_to_int[meta] for meta in metadata]
        )

        if fabric.global_rank == 0 and (((k + 1) <= 5) or ((k + 1) % 5 == 0)):
            fabric.print(f"Validation step {k+1} / {max_validation_steps}")

    print(f"Validation forward passes complete on rank ({fabric.global_rank}/{fabric.world_size})")
    # clean finish of the validation forwards
    fabric.barrier()
    if fabric.global_rank == 0:
        print("Validation forwards complete on all ranks.")

    # # Quick iteration check across all ranks, since logic below relies on this
    # local_total_steps_taken = torch.tensor(total_steps_taken, device=fabric.device)
    # gathered_total_steps_taken = fabric.all_gather(local_total_steps_taken).tolist()
    # if fabric.global_rank == 0:
    #     assert all(
    #         [step == total_steps_taken for step in gathered_total_steps_taken]
    #     ), "Validation steps taken each rank mismatch"
    #     print(f"All ranks took {total_steps_taken} validation steps.")

    if cfg.memorization_validation:
        total_em_stats, ds_wise_em_stats = litgpt.utils.reduce_memorization_metrics(
            em_stats_list=em_stats_list,
            total_steps_taken=total_steps_taken,
            fabric=fabric,
            cfg=cfg,
        )
    else:
        total_em_stats = None
        ds_wise_em_stats = None

    # gather individual val loss
    metadata_list = fabric.all_gather(metadata_list).reshape(-1).tolist()
    metadata_list = [ds_meta_from_int[meta] for meta in metadata_list]
    sample_wise_losses = fabric.all_gather(sample_wise_losses).reshape(-1).tolist()
    sums_and_counts = {}
    for category, value in zip(metadata_list, sample_wise_losses):
        if category not in sums_and_counts:
            sums_and_counts[category] = {"sum": 0, "count": 0}
        sums_and_counts[category]["sum"] += value
        sums_and_counts[category]["count"] += 1
    ds_wise_losses = {
        "ds_wise_losses/" + category: data["sum"] / data["count"] for category, data in sums_and_counts.items()
    }

    global_val_losses = fabric.all_gather(losses).reshape(-1)

    losses = losses[losses != -1.0]  # ignore filler elms
    local_val_loss = losses.mean().item()

    global_val_losses = global_val_losses[global_val_losses != -1.0]  # ignore filler elms
    global_val_loss = global_val_losses.mean().item()

    if correct_predictions is not None:
        global_correct_predictions = fabric.all_gather(correct_predictions).reshape(-1)

        local_correct_predictions = correct_predictions[correct_predictions != -1.0]
        local_accuracy = local_correct_predictions.mean().item()

        global_correct_predictions = global_correct_predictions[
            global_correct_predictions != -1.0
        ]  # ignore filler elms
        global_accuracy = global_correct_predictions.mean().item()

    model.train()
    if orig_objective is not None:
        model.set_objective(orig_objective)
        return (
            local_val_loss,
            global_val_loss,
            dict(total_em_stats=total_em_stats, ds_wise_em_stats=ds_wise_em_stats),
            local_accuracy,
            global_accuracy,
            ds_wise_losses,
        )

    torch.cuda.empty_cache()
    fabric.print(f"Manual cuda empty cache triggered!")
    gc.collect()
    fabric.print(f"Manual python gc triggered!")

    return (
        local_val_loss,
        global_val_loss,
        dict(total_em_stats=total_em_stats, ds_wise_em_stats=ds_wise_em_stats),
        ds_wise_losses,
    )


def train_step(input_ids, labels, fabric, state, running_loss, cfg):
    """Separate scope for a single train step, encapsulating the part that is actual work"""
    # Do some checks on the val loop and the throughput of the model.
    model = state["model"]
    optimizer = state["optimizer"]
    data_scheduler = state["data_scheduler"]
    val_data_scheduler = state["val_data_scheduler"]
    tokenizer = state["tokenizer"]
    metrics = {}

    state["microbatch_step"] += 1

    # Realize the input and labels tensors.
    input_ids = input_ids.to(fabric.device, non_blocking=True)
    labels = labels.to(fabric.device, non_blocking=True)
    mask, positions = get_attention_mask(input_ids, tokenizer, cfg.cache_attn, cfg.doc_block_attn)

    if state["microbatch_step"] < cfg.shape_watching_iters:
        bsz, seq_len = input_ids.shape
        fabric.print(f"bsz: {bsz} | seq_len: {seq_len}")
        fabric.print(f"input_ids.shape: {input_ids.shape} | labels.shape: {labels.shape}")
    elif state["microbatch_step"] == cfg.shape_watching_iters and cfg.shape_watching_iters > 0:
        fabric.print("Silencing shape watching ...")

    # Forward, loss, and backward computation.
    is_accumulating = state["microbatch_step"] % cfg.gradient_accumulation_steps != 0
    monitor_step = cfg.model_telemetry and state["microbatch_step"] % cfg.log_iter_interval == 0
    if monitor_step and not is_accumulating:
        model.module.apply(enable_monitoring_on_step)

    with fabric.no_backward_sync(model, enabled=is_accumulating):
        outputs = model(input_ids, position_ids=positions, labels=labels, attention_mask=mask)
        fabric.backward(outputs["loss"] / cfg.gradient_accumulation_steps, model=model)

    if not cfg.allow_nonfinite_loss and not torch.isfinite(outputs["loss"]):
        raise ValueError(f"Loss is {outputs['loss']} on {socket.gethostname()}. Terminating ...")
    metrics["mbs_loss"] = outputs["loss"].detach()
    running_loss.update(outputs["loss"].detach())

    # Take an optimization step if not accumulating.
    if not is_accumulating:
        metrics["grad_norm"] = fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_clip).detach()
        optimizer.step()
        if monitor_step:
            track_gradient_metrics(model, optimizer, metrics)
            model.module.apply(partial(disable_monitoring_and_retrieve_metrics, metrics=metrics))
        if cfg.fabric.use_apex_adamw:
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        state["optimizer_step"] += 1
        # Update learning rate (post-increment since we init it before the first step).
        next_step_lr = get_lr(it=state["microbatch_step"], lr_decay_iters=cfg.max_steps, cfg=cfg)

        # note we are first logging the base lr, not the scaled lr
        metrics["lr"] = next_step_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = next_step_lr if not param_group.get("scaler") else param_group["scaler"](next_step_lr)
            metrics[f"lr_{param_group['group_name']}"] = param_group["lr"]
        data_scheduler.step(state["optimizer_step"])
    else:
        metrics["grad_norm"] = None
        metrics["lr"] = None
    return metrics, is_accumulating


def train(
    fabric,
    state,
    train_dataloader,
    val_dataloader,
    cfg,
    *,
    resume_ckpt=None,
    data_scheduler: DataScheduler,
    val_data_scheduler: DataScheduler,
):
    """The main training loop."""

    first_validation_passed = False
    logged_time_to_first_validation = False

    if cfg.sanity_validate and not cfg.validate_only:
        validate(
            fabric,
            state["model"],
            val_dataloader,
            val_data_scheduler,
            max_validation_steps=2,
            tokenizer=state["tokenizer"],
            cfg=cfg,
            train_step=state["optimizer_step"],
        )
        first_validation_passed = True

    initial_iter = state["microbatch_step"]
    train_iterator = iter(train_dataloader)

    # Resume data loader state by fast-forwarding through all seen batches.
    # If we migrate to the streaming dataset in future, we might not need this.
    if resume_ckpt and not cfg.validate_only:
        resume_t0 = time.time()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")

            data_scheduler.step(resume_iter + 1)

        fabric.barrier()
        fabric.print(f"Resuming data loader finished. Took {time.time() - resume_t0:.1f} seconds to reach iteration")

    if cfg.initial_validate:
        t0 = time.time()
        val_results = validate(
            fabric,
            state["model"],
            val_dataloader,
            val_data_scheduler,
            cfg.eval_iters,
            state["tokenizer"],
            cfg=cfg,
            train_step=state["optimizer_step"],
        )
        if cfg.target_range_val is not None:
            local_val_loss, global_val_loss, val_em_stats, local_accuracy, global_accuracy, ds_wise_losses = val_results
        else:
            local_val_loss, global_val_loss, val_em_stats, ds_wise_losses = val_results
        td = time.time() - t0

        metrics = {
            "val_loss": global_val_loss,
            "val_ppl": math.exp(global_val_loss),
            "local_val_loss": local_val_loss,
            "local_val_ppl": math.exp(local_val_loss),
            "val_time": td,
            "microbatch_step": state["microbatch_step"],
            "optimizer_step": state["optimizer_step"],
        }
        if cfg.target_range_val is not None:
            metrics |= {
                "val_accuracy": global_accuracy,
                "local_val_accuracy": local_accuracy,
            }
        fabric.print(
            f"iter {state['microbatch_step']}: val loss {global_val_loss:.4f}, "
            f"{f'val acc {global_accuracy:.2f}, ' if cfg.target_range_val is not None else ''}",
            f"val time: {td * 1000:.2f} ms",
        )

        # also re-log any key in state that contains "data_scheduler_" in the key
        # since this gets all the stats from the most recent log_iter_interval but avoids
        # sending the actual data_scheduler object to the logger
        for k in state.keys():
            if "data_scheduler_" in k:
                metrics[k] = state[k]

        # em_stats is nested dict. wandb allows us to log metrics like key1/key2/key3: value
        # so we want to flatten the nested dict into a single level dict auto generating the flat keys
        if val_em_stats is not None:
            # print(f"val_em_stats: {json.dumps(val_em_stats, indent=4)}")
            flat_em_stats = litgpt.utils.flatten_dict(val_em_stats, sep="/", coerce_to_str=True)
            metrics.update(flat_em_stats)

        if ds_wise_losses is not None:
            metrics.update(ds_wise_losses)

        fabric.log_dict(metrics, step=state["microbatch_step"])
        first_validation_passed = True
        fabric.barrier()

    # Set up global loss monitor.
    running_loss = RunningMean(window=cfg.gradient_accumulation_steps, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.time()

    lr = get_lr(it=state["microbatch_step"], lr_decay_iters=cfg.max_steps, cfg=cfg)
    for param_group in state["optimizer"].param_groups:
        param_group["lr"] = lr if not param_group.get("scaler") else param_group["scaler"](lr)

    # Main training loop.
    step_time = 0
    while state["microbatch_step"] <= cfg.max_steps:
        # measure average time over last log_iter steps,
        # including the time to get the next batch.
        t0 = time.time()
        try:
            data_batch = next(train_iterator)
        except StopIteration:
            break

        is_accumulating = False
        if not cfg.validate_only:
            # Main work

            # input_ids, labels = data_batch
            input_ids, labels, metadata = data_batch

            metrics, is_accumulating = train_step(input_ids, labels, fabric, state, running_loss, cfg=cfg)
            step_time += time.time() - t0
            # Log at an interval.
            if state["microbatch_step"] % cfg.log_iter_interval == 0:
                log_iter(fabric, state, running_loss, initial_iter, total_t0, step_time, metrics, data_scheduler, cfg)
                step_time = 0

            # Maybe save
            maybe_save_checkpoint(fabric, state, cfg, is_accumulating=is_accumulating)

        if first_validation_passed and not logged_time_to_first_validation:
            # If a validation happened before the first train step, all potential compilation calls have resolved.
            fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
            logged_first_validation = True

        # Maybe validate
        validate_regular = not is_accumulating and state["optimizer_step"] % cfg.eval_step_interval == 0
        validate_at_the_end = state["microbatch_step"] >= cfg.max_steps - 1
        if validate_regular or validate_at_the_end or cfg.validate_only:
            t0 = time.time()
            val_results = validate(
                fabric,
                state["model"],
                val_dataloader,
                val_data_scheduler,
                cfg.eval_iters,
                state["tokenizer"],
                cfg=cfg,
                train_step=state["optimizer_step"],
            )
            if cfg.target_range_val is not None:
                local_val_loss, global_val_loss, val_em_stats, local_accuracy, global_accuracy, ds_wise_losses = (
                    val_results
                )
            else:
                local_val_loss, global_val_loss, val_em_stats, ds_wise_losses = val_results
            td = time.time() - t0

            metrics = {
                "val_loss": global_val_loss,
                "val_ppl": math.exp(global_val_loss),
                "local_val_loss": local_val_loss,
                "local_val_ppl": math.exp(local_val_loss),
                "val_time": td,
                "microbatch_step": state["microbatch_step"],
                "optimizer_step": state["optimizer_step"],
            }
            if cfg.target_range_val is not None:
                metrics |= {
                    "val_accuracy": global_accuracy,
                    "local_val_accuracy": local_accuracy,
                }
            fabric.print(
                f"iter {state['microbatch_step']}: val loss {global_val_loss:.4f}, "
                f"{f'val acc {global_accuracy:.2f}, ' if cfg.target_range_val is not None else ''}",
                f"val time: {td * 1000:.2f} ms",
            )

            # also re-log any key in state that contains "data_scheduler_" in the key
            # since this gets all the stats from the most recent log_iter_interval but avoids
            # sending the actual data_scheduler object to the logger
            for k in state.keys():
                if "data_scheduler_" in k:
                    metrics[k] = state[k]

            # em_stats is nested dict. wandb allows us to log metrics like key1/key2/key3: value
            # so we want to flatten the nested dict into a single level dict auto generating the flat keys
            if val_em_stats is not None:
                # print(f"val_em_stats: {json.dumps(val_em_stats, indent=4)}")
                flat_em_stats = litgpt.utils.flatten_dict(val_em_stats, sep="/", coerce_to_str=True)
                metrics.update(flat_em_stats)

            if ds_wise_losses is not None:
                metrics.update(ds_wise_losses)

            fabric.log_dict(metrics, step=state["microbatch_step"])

            first_validation_passed = True
            if not logged_time_to_first_validation:
                # This is another moment that all potential compilation calls may have resolved
                fabric.log_to_summary({"first_validation_passed": time.time() - global_start_time})
                logged_time_to_first_validation = True

            fabric.barrier()

        if cfg.validate_only:
            break

        if state["microbatch_step"] >= cfg.max_steps - 1:
            break


####################################################################################################
# Train loop sub-routines.
####################################################################################################


def log_iter(
    fabric: Fabric,
    state: dict,
    running_loss: RunningMean,
    initial_iter: int,
    total_t0: float,
    accumulated_step_time: float,
    metrics: dict,
    data_scheduler: Optional[DataScheduler],
    cfg: CLISettings,
):
    """Log the iteration and compute the throughput."""
    loss = running_loss.compute()
    t1 = time.time()

    # Log additional metrics.
    metrics = {} if metrics is None else metrics

    avg_time_per_step = accumulated_step_time / cfg.log_iter_interval

    if cfg.fabric_strategy == "axonn_tp":
        from axonn import axonn as ax

        batch_size_1 = cfg.micro_batch_size * ax.config.G_intra_d * ax.config.G_data
        batch_size_2 = cfg.micro_batch_size * fabric.world_size / ax.config.G_intra_r
        assert batch_size_1 == batch_size_2, "batch_size_1 != batch_size_2"
        tokens_per_step = batch_size_1 * cfg.block_size
    else:
        tokens_per_step = cfg.micro_batch_size * cfg.block_size * fabric.world_size

    tokens_per_second = tokens_per_step / avg_time_per_step

    # derive some frontier costs
    if cfg.derive_cost_basis:
        assert state["microbatch_step"] == state["optimizer_step"], "cost calc assumes no accumulation"
        # max steps is always derived even from max tokens so use this
        remaining_steps = cfg.max_steps - state["microbatch_step"]
        # we assume we maintain the same throughput for the remaining steps
        remaining_time = remaining_steps * avg_time_per_step  # seconds
        # we can estimate the total cost of the job in node hours and wall clock hours
        estimated_total_time = (t1 - total_t0) + remaining_time  # seconds

        remaining_node_hours = (remaining_time / 3600) * cfg.num_nodes
        remaining_wall_hours = remaining_time / 3600
        total_node_hours = (estimated_total_time / 3600) * cfg.num_nodes
        total_wall_hours = estimated_total_time / 3600
        metrics |= {
            "cost_basis/remaining_node_hours": remaining_node_hours,
            "cost_basis/remaining_wall_hours": remaining_wall_hours,
            "cost_basis/total_node_hours": total_node_hours,
            "cost_basis/total_wall_hours": total_wall_hours,
            "cost_basis/max_steps": cfg.max_steps,
            "cost_basis/remaining_steps": remaining_steps,
        }

    # log a "were stable" flag we can filter for
    if cfg.stability_step is not None:
        if state["microbatch_step"] >= cfg.stability_step:
            metrics |= {"run_is_stable": True}
        else:
            metrics |= {"run_is_stable": False}

    metrics |= {
        "local_loss": loss.clone().detach(),
        "local_ppl": loss.exp(),
        "microbatch_step": state["microbatch_step"],
        "optimizer_step": state["optimizer_step"],
        "steps_per_second": 1 / avg_time_per_step,
        "seconds_per_step": avg_time_per_step,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_per_gpu": tokens_per_second / fabric.world_size,
        "remaining_time": (
            (t1 - total_t0) / (state["microbatch_step"] - initial_iter) * (cfg.max_steps - state["microbatch_step"])
        ),
        "total_tokens": state["microbatch_step"] * tokens_per_step,
        "total_time": t1 - total_t0,
    }
    if cfg.measure_utilization:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_memory_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        model_flops, tflops, mfu = get_MFU_metrics(tokens_per_second, fabric, state["model"], cfg.fabric_precision)
        metrics |= {
            "total_FLOPs": state["microbatch_step"] * tokens_per_step * model_flops,
            "TFLOPs": tflops,
            "model_flop_utilization": mfu,
            "max_mem_allocated_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_memory_reserved_per_gpu,
        }
    if cfg.simple_gptneox_tflops:
        max_memory_allocated_per_gpu = torch.cuda.max_memory_allocated(fabric.device) / 1024**3
        max_memory_reserved_per_gpu = torch.cuda.max_memory_reserved(fabric.device) / 1024**3
        torch.cuda.reset_peak_memory_stats(fabric.device)
        if cfg.fabric_strategy == "axonn_tp":
            # try various ways of counting batch size
            from axonn import axonn as ax

            batch_size = cfg.micro_batch_size * ax.config.G_intra_d * ax.config.G_data
            neox_flops = simple_gptneox_tflops(
                metrics, fabric, cfg, batch_size=batch_size, iter_time_s=metrics["seconds_per_step"]
            )
            # fabric.print(f"simple_gptneox_tflops(micro_bs*depth*data): {neox_flops:4.2f}")

            simple_flops = neox_flops
        else:
            simple_flops = simple_gptneox_tflops(metrics, fabric, cfg, iter_time_s=metrics["seconds_per_step"])
        simple_mfu = simple_flops / cfg.peak_tflops_per_device
        metrics |= {
            "TFLOPs": simple_flops,
            "model_flop_utilization": simple_mfu,
            "max_mem_allocated_per_gpu": max_memory_allocated_per_gpu,
            "max_mem_reserved_per_gpu": max_memory_reserved_per_gpu,
        }

    # Update loss and grad_norm with all_reduce
    # FIXME _these_ could be expensive if the topo is large, so do we need to always report
    # world reduced loss or is rank-local loss sufficient? Maybe add a flag option.
    if metrics["grad_norm"] is not None:
        grad_norm = fabric.all_reduce(metrics["grad_norm"])
    metrics["global_loss"] = fabric.all_reduce(loss)
    metrics["global_train_ppl"] = metrics["global_loss"].exp()
    metrics["global_grad_norm"] = grad_norm

    if data_scheduler is not None:
        curr_data_weights = data_scheduler.get_data_weights()
        curr_data_weights = dict(zip(cfg.dataset_names, curr_data_weights))

        curr_sample_count = data_scheduler.get_sample_count()
        curr_sample_count = fabric.all_reduce(curr_sample_count, reduce_op="sum")

        curr_epoch_count = data_scheduler.get_epoch_count()
        curr_epoch_count = fabric.all_reduce(curr_epoch_count, reduce_op="mean")

        for i, x in enumerate(curr_data_weights.keys()):
            metrics["data_scheduler_weight/" + x] = curr_data_weights[x]
            metrics["data_scheduler_norm_weight/" + x] = curr_data_weights[x] / sum(list(curr_data_weights.values()))
            metrics["data_scheduler_sample_count/" + x] = curr_sample_count[i]
            metrics["data_scheduler_epoch_count/" + x] = curr_epoch_count[i]

            state["data_scheduler_weight/" + x] = metrics["data_scheduler_weight/" + x]
            state["data_scheduler_norm_weight/" + x] = metrics["data_scheduler_norm_weight/" + x]
            state["data_scheduler_sample_count/" + x] = metrics["data_scheduler_sample_count/" + x]
            state["data_scheduler_epoch_count/" + x] = metrics["data_scheduler_epoch_count/" + x]

    fabric.log_dict(metrics, step=state["microbatch_step"])

    # log some important overall metrics as summary
    if cfg.log_scaling_law_metrics:
        fabric.log_to_summary(
            {
                "scaling/global_loss": metrics["global_loss"],
            }
        )

    # Log some metrics to the console.
    step_timing = (
        f" steps/sec: {metrics['steps_per_second']:.2f} |\n"
        if metrics["steps_per_second"] >= 1.0
        else f" secs/step: {metrics['seconds_per_step']:.2f} |\n"
    )
    fabric.print(
        f"Iteration {metrics['microbatch_step']:>6} | Loss: {metrics['global_loss']:7.4f} | {metrics['global_train_ppl']:4.2f} PPL      |"
        f" Update {metrics['optimizer_step']:>6}|\n"
        f"(optimizer.step) "
        f"| MFU : {metrics.get('model_flop_utilization', 0):6.2%}  | TFLOPs : {metrics.get('TFLOPs', 0):4.2f} "
        f"| Max mem (alloc/reserved): {metrics.get('max_mem_allocated_per_gpu', 0):.2f}/{metrics.get('max_mem_reserved_per_gpu', 0):.2f} GB |"
        f" tok/sec: {metrics['tokens_per_second']:8.1f} |  tok/sec/gpu: {metrics['tokens_per_second_per_gpu']:8.1f} |"
        f"{step_timing}"
        f"                 | LR: {metrics['lr']:2.4e}| Grad norm: {metrics['global_grad_norm']:6.4f} |\n"
        f"                 | Tokens: {metrics['total_tokens']/1e9: 4.1f}B | exaFLOP: {metrics.get('total_FLOPs', 0) / 1e18:8.5f} |"
        f" Remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days |"
    )


####################################################################################################
# Data utility functions.
####################################################################################################


def create_dataloader(
    data_config: list[litgpt.settings.DataEntry],
    batch_size: int,
    block_size: int,
    n_chunks: int,
    data_dir: str,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
) -> tuple[DataLoader, DataSchedulerTracker]:
    global_data_dir = data_dir
    datasets = []
    for curr_config in data_config:

        if curr_config.type == "hfds":
            assert tokenizer is not None, "tokenizer must be provided for HuggingfaceDataset"
            assert curr_config.data_dir is not None, "data_dir must be provided for HuggingfaceDataset"
            dataset = HuggingfaceDataset(
                ds_name_or_path=curr_config.data_dir,  # this is a path to a previously save_to_disk'd hfds
                seed=seed,
                num_processes=(
                    fabric.world_size
                    if not cfg.fabric_strategy == "axonn_tp"
                    else fabric.strategy.global_world_size_for_creating_dataloader()
                ),
                process_rank=(
                    fabric.global_rank
                    if not cfg.fabric_strategy == "axonn_tp"
                    else fabric.strategy.global_rank_for_creating_dataloader()
                ),
                data_id=curr_config.prefix,  # this is provided for logging, and schedule purposes
                return_data_id=curr_config.return_data_id
                or cfg.return_data_id,  # this is returned to manage rows dynamically
                data_signature=curr_config.data_signature or cfg.data_signature,  # specification of the data fmt
                repetitions=curr_config.repetitions,  # repeat the dataset a number of times
            )

        elif curr_config.type == "pkds":
            prefix = curr_config.prefix

            if curr_config.data_dir is not None:
                data_dir = curr_config.data_dir
            else:
                data_dir = global_data_dir

            if fabric.global_rank == 0:
                filenames = [str(pth) for pth in sorted(Path(data_dir).glob(f"{prefix}*"))]
                if cfg.shuffle_filenames:
                    random.seed(seed)
                    random.shuffle(filenames)  # inplace
                if not filenames:
                    raise FileNotFoundError(f"No files found at {str(data_dir)} with prefix {prefix}.")
            else:
                filenames: list[str] = None  # type: ignore # hashtag believe

            filenames = fabric.broadcast(filenames, 0)  # this is a blocking op from rank 0 to all other ranks

            # log after broadcast so we know we passed it.
            if fabric.global_rank == 0:
                num_processes = (fabric.world_size,)
                process_rank = (fabric.global_rank,)
                fabric.print(
                    f"Rank ({process_rank}/{num_processes}) glob'd {len(filenames)} files"
                    f" from {data_dir}{f' w/ prefix {prefix}' if prefix not in ['','*'] else ''},"
                    f" files[:3]: {filenames[:3]}"
                )

            dataset = PackedDataset(
                filenames,
                n_chunks=n_chunks,
                block_size=block_size,
                shuffle=cfg.shuffle_blocks,
                seed=seed,
                num_processes=(
                    fabric.world_size
                    if not cfg.fabric_strategy == "axonn_tp"
                    else fabric.strategy.global_world_size_for_creating_dataloader()
                ),
                process_rank=(
                    fabric.global_rank
                    if not cfg.fabric_strategy == "axonn_tp"
                    else fabric.strategy.global_rank_for_creating_dataloader()
                ),
                data_id=prefix,
                return_data_id=curr_config.return_data_id
                or cfg.return_data_id,  # this is returned to manage rows dynamically
            )
        elif curr_config.type == "rngds":
            # Debugging option
            generator = torch.Generator()
            generator.manual_seed(seed)
            dataset = torch.randint(
                0,
                tokenizer.vocab_size,
                (int(1e6), block_size),
                dtype=torch.int32,
                generator=generator,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {curr_config.type}")

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [curr_config.weight for curr_config in data_config]
    data_scheduler_tracker = DataSchedulerTracker(weights)

    combined_dataset = CombinedDataset(
        datasets=datasets, seed=seed, data_scheduler_tracker=data_scheduler_tracker, data_telemetry=cfg.data_telemetry
    )

    parametrized_collate_fn = partial(
        generic_collate_fn,
        tokenizer=tokenizer,
        block_size=cfg.loader_block_size,
        pad_to_block_size=cfg.pad_to_block_size,
        add_bos=cfg.add_bos,
        add_eos=cfg.add_eos,
        collate_checks_enabled=cfg.collate_checks_enabled,
        all_block_size_tensors=cfg.all_block_size_tensors,
    )

    return (
        DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=parametrized_collate_fn,
            num_workers=cfg.dataloader_num_workers,
        ),
        data_scheduler_tracker,
    )


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: Fabric,
    seed: int = 1337,
    *,
    cfg: CLISettings,
    tokenizer: Tokenizer,
) -> Tuple[DataLoader, Optional[DataLoader], DataSchedulerTracker]:

    cfg.train_dataset_prefixes = [ds.prefix for ds in cfg.data_config["train_data"]]
    cfg.val_dataset_prefixes = (
        [ds.prefix for ds in cfg.data_config["val_data"]] if "val_data" in cfg.data_config else []
    )

    fabric.print(f"Creating dataloaders with seed: {seed}")
    train_dataloader, data_scheduler_tracker = create_dataloader(
        cfg.data_config["train_data"],
        batch_size=batch_size,
        block_size=block_size,
        n_chunks=cfg.n_chunks,
        fabric=fabric,
        data_dir=cfg.train_data_dir,
        seed=seed,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    val_dataloader, val_data_scheduler_tracker = (
        create_dataloader(
            cfg.data_config["val_data"],
            batch_size=batch_size,
            block_size=block_size,
            n_chunks=cfg.n_chunks,
            fabric=fabric,
            data_dir=cfg.val_data_dir,
            seed=seed,
            cfg=cfg,
            tokenizer=tokenizer,
        )
        if "val_data" in cfg.data_config
        else (None, None)
    )
    return train_dataloader, val_dataloader, data_scheduler_tracker, val_data_scheduler_tracker


####################################################################################################
# Train utility functions.
####################################################################################################


def derive_precision(precision, strategy_details):
    """ "Precision setup for torch fsdp"""
    import torch.distributed.fsdp

    param_dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "16" in precision else torch.float32
    reduce_dtype = torch.float32 if "mixed" in precision else param_dtype
    if r := strategy_details.all_reduce_dtype is not None:
        reduce_dtype = (
            torch.float16
            if r in ["16", "fp16", "fp16-mixed"]
            else torch.bfloat16 if r in ["bf16", "bf16-mixed"] else torch.float32
        )
    return torch.distributed.fsdp.MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        # cast_forward_inputs=False,
    )


def get_attention_mask(input_ids, tokenizer, cache_attn=True, doc_block_attn=True):
    mask, position_ids = None, None
    if doc_block_attn:
        mask, position_ids = get_ltor_masks_and_position_ids(
            input_ids, tokenizer.eod_token_id, reset_position_ids=True, reset_attention_mask=True
        )
    elif cache_attn:
        mask, position_ids = get_cache_attn_masks(
            input_ids, tokenizer.cache_token_id, reset_position_ids=True, reset_attention_mask=True
        )
    return mask, position_ids


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int, lr_decay_iters: int, cfg: CLISettings) -> float:
    assert lr_decay_iters == cfg.max_steps, "lr_decay_iters must be equal to max_steps for curr logic."
    # 0) compute cooldown start and decay ratio
    if cfg.cooldown_iters > 0:
        # add extra + 1 to cooldown_iters below to actually hit 0.0 over a full cfg.cooldown_iters steps
        # but current choice is to not realize/use the last "0.0" lr
        cooldown_start = lr_decay_iters - (cfg.cooldown_iters)
        total_decay_steps = lr_decay_iters - cfg.warmup_iters - (cfg.cooldown_iters)
        decay_ratio = (it - cfg.warmup_iters) / (total_decay_steps)
    else:
        cooldown_start = lr_decay_iters + 1  # should never hit the cooldown block then
        non_decay_steps = cfg.warmup_iters
        decay_ratio = (it - non_decay_steps) / (lr_decay_iters - non_decay_steps)  # equiv to orig
    base_lr = cfg.optim_config["lr"]
    min_lr = cfg.min_lr

    # 1) if in linear warmup region
    if it < cfg.warmup_iters:
        return base_lr * it / cfg.warmup_iters
    # 2) if in linear cooldown region
    if it >= cooldown_start:
        cooldown_ratio = 1 - (it - cooldown_start) / (cfg.cooldown_iters)
        if cfg.lr_schedule in ["linear", "cosine"]:
            # we cool from min_lr to 0.0
            return max(min_lr * cooldown_ratio, 0.0)
        else:  # eg. "constant" or "trapezoid"
            # we linearly cool, but never below min_lr
            return max(base_lr * cooldown_ratio, min_lr)
    # X) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        raise ValueError(f"it={it} is greater than lr_decay_iters={lr_decay_iters}, weird.")
        return min_lr
    # 3) in between, decay from base_lr down to min_lr
    assert 0 <= decay_ratio <= 1
    if cfg.lr_schedule == "linear":
        return base_lr - decay_ratio * (base_lr - min_lr)
    elif cfg.lr_schedule in ["constant", "trapezoid"]:
        return base_lr
    elif cfg.lr_schedule == "cosine":
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (base_lr - min_lr)
    else:
        raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def load_checkpoint(
    fabric, state, out_dir, run_name, model_checkpoint, model_impl="litgpt", resume=True, fabric_strategy=None
):
    resume_ckpt = None
    if resume:
        base_for_glob = Path(out_dir) / fabric.get_prefix_for_checkpoint()
        fabric.print(f"Globbing for checkpoint files in {base_for_glob}")
        if fabric_strategy == "axonn_tp":
            ckpt_pattern = f"*/*-{run_name}.pth"
        else:
            ckpt_pattern = f"*-{run_name}.pth"
        ckpt_paths = list(base_for_glob.glob(ckpt_pattern))
        if len(ckpt_paths) == 0:
            fabric.print(f"No checkpoint found in {out_dir} to resume from.")
        else:
            resume_ckpt = max(
                ckpt_paths,
                key=(lambda p: int(p.name.split("-")[1].split(f"-{run_name}.pth")[0])),
            )
            filename, directory = str(resume_ckpt.name), resume_ckpt.parents[0]
            filename = filename[filename.find("step") :]
            # FIXME, with current api, we reove the inner dir structure to pretend we dont know about it
            if fabric_strategy == "axonn_tp":
                directory = Path(out_dir) / fabric.get_prefix_for_checkpoint()
            resume_ckpt = directory / filename
            fabric.print(f"Resuming training from {resume_ckpt}")
            fabric.load(resume_ckpt, state)

    if resume_ckpt is None and model_checkpoint is not None:
        if model_impl == "litgpt":
            checkpoint_path = f"{model_checkpoint}/lit_model.pth"
        elif model_impl == "dynamic":
            checkpoint_path = f"{model_checkpoint}/lit_model_dynamic.pth"
        else:
            raise ValueError(f"Invalid checkpoint loader for model implementation {model_impl}.")
        fabric.print(f"Loading pretrained model checkpoint from {checkpoint_path}")
        litgpt.utils.load_checkpoint(fabric, state["model"], checkpoint_path)
    return resume_ckpt


def maybe_save_checkpoint(fabric, state, cfg, is_accumulating=False, force_save=False):
    # Pathing for various save conditions.
    prefix = fabric.get_prefix_for_checkpoint()
    fully_qualified_checkpoint_path = f"{cfg.out_dir}/{prefix}/step-{state['optimizer_step']:08d}-{cfg.run_name}.pth"

    # Check the three save conditions:
    save_at_interval = not is_accumulating and state["optimizer_step"] % cfg.save_step_interval == 0
    if cfg.save_n_min_before_job_done is not None:
        time_spent = time.time() - global_start_time
        remaining_time = cfg.global_total_time - time_spent
        remaining_time = remaining_time / 60.0
        remaining_time = fabric.all_reduce(remaining_time, reduce_op="mean")
        save_before_timeout = remaining_time <= cfg.save_n_min_before_job_done
        if save_before_timeout:
            fabric.print(f"Saving at {remaining_time:.02f} minutes left")
            cfg.save_n_min_before_job_done = None  # reset
    else:
        save_before_timeout = False
    save_at_last_step = cfg.save_last_step and (state["microbatch_step"] >= (cfg.max_steps - 1))

    if save_at_interval or save_at_last_step or save_before_timeout or force_save:
        fabric.print(f"Saving checkpoint to {str(fully_qualified_checkpoint_path)!r}")
        fabric.save(fully_qualified_checkpoint_path, state)


def _get_time_from_slurm() -> int:
    try:
        global_total_str_parse = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()  # this is slow
        global_total_str_parse = global_total_str_parse.strip("\n")
        global_total_str_parse = [int(i) for i in re.split(":|-", global_total_str_parse)]
        if len(global_total_str_parse) == 4:
            global_total_time = (
                24 * 3600 * global_total_str_parse[0]
                + 3600 * global_total_str_parse[1]
                + 60 * global_total_str_parse[2]
                + global_total_str_parse[3]
            )
        elif len(global_total_str_parse) == 3:
            global_total_time = (
                3600 * global_total_str_parse[0] + 60 * global_total_str_parse[1] + global_total_str_parse[2]
            )
        elif len(global_total_str_parse) == 2:
            global_total_time = 60 * global_total_str_parse[0] + global_total_str_parse[1]
    except Exception as e:
        print(e)
        global_total_time = 9999999999999999
    return global_total_time


####################################################################################################
# Main control loop
####################################################################################################
import sys
import datetime


def main():
    """Encapsulates main scope away from import calls."""

    # Configuration loader
    cfg: CLISettings = CLI(CLISettings)  # type: ignore

    # Print system setup
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print("--------------------------------------------------------------------")
        print(f"------------------ Launching run {cfg.run_name}------------------")
        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")
        print(f"Platform: {sys.platform}, Python: {sys.version.split(' (')[0]}, PyTorch: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")
        print(f"GPU : {DEVICE_NAME}. DRIVER: {DEVICE_DRIVER_VERSION}.")

    set_torch_flags(cfg)  # should come before fabric setup
    # Next we set up the fabric and logger.
    fabric = setup_fabric(cfg)

    # Now we call the main function with the fabric and cfg.
    state, train_dataloader, val_dataloader, data_scheduler, val_data_scheduler, resume_ckpt = startup(fabric, cfg)

    # In cases such as training from scratch, saving an initial checkpoint could be desired.
    if cfg.save_first_step:
        maybe_save_checkpoint(fabric, state, cfg, force_save=True)

    # Now we call the train function with the fabric, state, and dataloaders.
    train_time = time.time()
    train(
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        cfg,
        resume_ckpt=resume_ckpt,
        data_scheduler=data_scheduler,
        val_data_scheduler=val_data_scheduler,
    )

    if sum(data_scheduler.get_data_weights()) == 0:  # some extra validation if we exited on max_epochs
        fabric.barrier()
        val_results = validate(
            fabric,
            state["model"],
            val_dataloader,
            val_data_scheduler,
            cfg.eval_iters,
            state["tokenizer"],
            cfg=cfg,
            train_step=state["optimizer_step"],
        )
        if cfg.target_range_val is not None:
            raise NotImplementedError("Target range validation not implemented for max_epochs exit.")
        local_val_loss, global_val_loss, val_em_stats, ds_wise_losses = val_results
        fabric.print(f"iter {state['microbatch_step']}: val loss {global_val_loss:.4f}")
        if cfg.save_last_step:
            maybe_save_checkpoint(fabric, state, cfg, force_save=True)  # forcing a save as we are done!

    # Now exit
    fabric.print("--------------------------------------------------------------------")
    fabric.print(f"Training time: {str(datetime.timedelta(seconds=time.time() - train_time))} ")
    fabric.log_to_summary(
        {"train_time": time.time() - global_start_time, "total_time": time.time() - global_start_time}
    )
    if fabric.device.type == "cuda":
        max_alloc = f"{torch.cuda.max_memory_allocated(fabric.device)/float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(fabric.device)/float(1024**3):,.3f} GB"
        fabric.print(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    fabric.print("--------------------------------------------------------------------")
    if torch.distributed.is_initialized():
        # torch.distributed.barrier()  # this could be very good or very bad
        torch.distributed.destroy_process_group()  # Force a clean exit
    if int(os.getenv("SLURM_PROCID", "0")) == 0:
        print(f"Run {cfg.run_name} finished without error.")
        print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
        print("-----------------Shutdown complete.--------------------------")


def guarded_main():
    try:
        main()
    except BaseException as e:  # gate around hell to guarantee NCCL deconstruction
        if torch.distributed.is_initialized():
            # torch.distributed.barrier()  # this could be very good or very bad
            torch.distributed.destroy_process_group()  # Force a clean exit
        if int(os.getenv("SLURM_PROCID", "0")) == 0:
            print("Run finished with errors.")
            print(f"---------Total time: {str(datetime.timedelta(seconds=time.time() - global_start_time))} ---------")
            print("-----------------Shutdown complete.--------------------------")

            raise


if __name__ == "__main__":
    guarded_main()

########## Misc Notes ######################

# 1)
# the lr schedule is computed as a function of iters not optim steps, but only evaluated after an optim step,
# so that the optim step lr lags a bit behind the current lr
# These are different if gradient_accumulation_steps > 1.
# There doesn't seem to be anything _incorrect_ about this, but it might
# not be very intuitive when picking schedule params.

# 2)
# unless prohibitively slow, we should be able to call the
# scripts.convert_pretrained_checkpoint.convert_checkpoint function in save_checkpoint
# which would turn the training checkpoint into a final saved model.
# Could even call the lit-to-hf conversion process as well.
# jog: can this be offloaded to a separate thread?

# 3)
# Saving and validating run on optimizer_step, while the main training loop runs
# on microbatch_step (microbatch steps) - this can be problematic if both are out of sync
# or if gradient accum frequency is not the right divisor
# and then learning rate, as above is on the MBS schedule
# couldn't we put everything on the mbs schedule?

# 4)
# No tokens should be added in train, this just mucks up the tokenizer internals and reproducibility,
# either some token (like <cache>) exists, or it does not. This should not be discovered/changed in train.py.
# Also we kill performance by doing 2**16+1 tokens. Tokenizer should be entirely constant

# 5)
# FIXME, token counting logic assumes fixed microbatch size w/ no padding.
# This is fine for pretraining style data, but this might not always be true.
