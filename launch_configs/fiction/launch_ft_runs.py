import os
import glob
import re
import json
from itertools import product, chain

# DRY_RUN = True
DRY_RUN = False

WRITE_ONLY = False
# WRITE_ONLY = True

DO_TRAIN = True
DO_CONVERT = False
DO_EVAL = False

# DO_TRAIN = False
# DO_CONVERT = True
# DO_EVAL = False

# DO_TRAIN = False
# DO_CONVERT = False
# DO_EVAL = True

# PUSH_TO_HUB = True
PUSH_TO_HUB = False

# EVAL_NODE_COUNT = None
EVAL_NODE_COUNT = 1

# EVAL_GPUS_PER_NODE = None
EVAL_GPUS_PER_NODE = 1


assert DO_TRAIN ^ (DO_CONVERT or DO_EVAL)

LAUNCHER_FILEPATH = "/p/lustre5/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = "/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.3.1/install/lib"

WRKSPC = os.getenv("WRKSPC")

WANDB_PROJECT_NAME = "fiction"
WANDB_OFFLINE = "False"
# WANDB_OFFLINE = "True"

BASE_OUT_DIR = f"/p/lustre5/kirchenb/llm-pretraining-root/lit-gpt-dev-fiction/output"

# QOS = "pdebug"
QOS = "pbatch"

JOB_LIMIT = None
# JOB_LIMIT = 1

# TIME_LIMIT = 59
# TIME_LIMIT = 119
TIME_LIMIT = 480


MAX_STEPS = 1500
EVAL_INTERVAL = 200
SAVE_INTERVAL = 200

# MAX_STEPS = 500
# EVAL_INTERVAL = 50
# SAVE_INTERVAL = 50


# for std fiction splits
EVAL_ITERS = 75  # (75 - (1500+3000+4100)/128)*128 = ~1000 base @ 128 wbsz

# for the base only control which checks all sets
# EVAL_ITERS = 145  # (145 - (1500*7+100+3000+4100)/128)*128 = ~1000 base @ 128 wbsz

LM_EVAL_BATCH_SIZE = 8

EVAL_STEPS = None
# EVAL_STEPS = [350, 400, 450, 499]
# EVAL_STEPS = [400, 450, 499]
# EVAL_STEPS = [499]

# EXP_VERSION = "v5"  # higher save granularity, but shorter
# EXP_VERSION = "v6"  # even higher save granularity, even shorter
# EXP_VERSION = "v7"  # just the base only control
EXP_VERSION = "v8"  # a redo of just doc split with a true val set
BASE_RUN_NAME = f"exp1_train_val_splits"

# ENVIRONMENT = "/usr/workspace/$USER/tuolumne_conda_28_630_fiction"  # or handle externally
PYTORCH_VERSION = "2.8"
ROCM_VERSION = "6.3.0"
RCCL_MODE = "rdzv"
# RCCL_MODE = "rdzv-lbann"
# RCCL_MODE = "eager"

pytorch_version_str = f'pt{PYTORCH_VERSION.replace(".", "")}'
rocm_version_str = f'rocm{ROCM_VERSION.replace(".", "")}'
TAG_LIST = [BASE_RUN_NAME, EXP_VERSION, pytorch_version_str, rocm_version_str, RCCL_MODE]
TAG_LIST = [str(t).replace(".", "") for t in TAG_LIST]
WANDB_TAG_STRING = f"[{','.join(TAG_LIST)}]"

# models
exp_list = [
    [
        4,
        8,
        128,
        2048,
        "True",
        "launch_configs/fiction/model_configs/tuolumne_llama-3-2-1B.json",
    ],
    [
        4,
        8,
        128,
        2048,
        "True",
        "launch_configs/fiction/model_configs/tuolumne_llama-3-2-3B.json",
    ],
    [
        4,
        8,
        128,
        2048,
        "True",
        "launch_configs/fiction/model_configs/tuolumne_llama-3-1-8B.json",
    ],
    [
        4,
        8,
        128,
        2048,
        "True",
        "launch_configs/fiction/model_configs/tuolumne_gemma-2b.json",
    ],
    [
        4,
        8,
        128,
        2048,
        "True",
        "launch_configs/fiction/model_configs/tuolumne_gemma-2-2b.json",
    ],
    # [
    #     8,
    #     4,  # 8 potentially too aggressive for this model w/ reduced code efficiency
    #     128,
    #     2048,
    #     "True",
    #     "launch_configs/fiction/model_configs/tuolumne_gemma-2-9b.json",
    # ],
]


# data
# pair train ds's with evaluation tasks
base_evals = ["fict_qa_obqa_blind_inf_ex_dedup_ds_mcq_topk4", "fict_qa_obqa_blind_inf_ex_dedup_ds_mcq_topk10"]
split_k_grid = list(product(["train", "val"], [4, 10]))
k_grid = [4, 10]

# 5 pct
BASE_RUN_NAME = f"exp1_train_val_splits_5pct"
sweep_hparam = [
    # [
    #     "launch_configs/fiction/data_configs/train_val_splits_5pct/tuolumne_event_split_fictions_train_val.json",
    #     ",".join(
    #         [f"event_split_fictions_webtext_{spl}_ds_valratio0.33_seed1234_mcq_topk{k}" for spl, k in split_k_grid]
    #         + base_evals
    #     ),
    # ],
    # [
    #     "launch_configs/fiction/data_configs/train_val_splits_5pct/tuolumne_event_split_fictsheets_train_val.json",
    #     ",".join(
    #         [f"event_split_fictsheets_webtext_{spl}_ds_valratio0.33_seed1234_mcq_topk{k}" for spl, k in split_k_grid]
    #         + base_evals
    #     ),
    # ],
    [
        "launch_configs/fiction/data_configs/train_val_splits_5pct/tuolumne_doc_split_train_val.json",
        ",".join(
            [
                f"style_strat_doc_split_fictions_{spl}_ds_valct1_styleNone_seed1234_mcq_topk{k}"
                for spl, k in split_k_grid
            ]
            + base_evals
        ),
    ],
    # [
    #     "launch_configs/fiction/data_configs/train_val_splits_5pct/tuolumne_style_split_loo_blog_train_val.json",
    #     ",".join(
    #         [
    #             f"style_strat_doc_split_fictions_{spl}_ds_valctNone_styleblog_seed1234_mcq_topk{k}"
    #             for spl, k in split_k_grid
    #         ]
    #         + base_evals
    #     ),
    # ],
    # [
    #     "launch_configs/fiction/data_configs/train_val_splits_5pct/tuolumne_style_split_loo_news_train_val.json",
    #     ",".join(
    #         [
    #             f"style_strat_doc_split_fictions_{spl}_ds_valctNone_stylenews_seed1234_mcq_topk{k}"
    #             for spl, k in split_k_grid
    #         ]
    #         + base_evals
    #     ),
    # ],
    # [
    #     "launch_configs/fiction/data_configs/train_val_splits_0pct/tuolumne_base_only_all_val.json",
    #     ",".join([f"fict_qa_obqa_blind_inf_ex_dedup_ds_mcq_topk{k}" for k in k_grid]),
    # ],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

# # 50 pct
# BASE_RUN_NAME = f"exp1_train_val_splits_50pct"
# sweep_hparam = [
#     [
#         "launch_configs/fiction/data_configs/train_val_splits_50pct/tuolumne_event_split_fictions_train_val.json",
#         ",".join(
#             [f"event_split_fictions_webtext_{spl}_ds_valratio0.33_seed1234_mcq_topk{k}" for spl, k in split_k_grid]
#             + base_evals
#         ),
#     ],
#     [
#         "launch_configs/fiction/data_configs/train_val_splits_50pct/tuolumne_event_split_fictsheets_train_val.json",
#         ",".join(
#             [f"event_split_fictsheets_webtext_{spl}_ds_valratio0.33_seed1234_mcq_topk{k}" for spl, k in split_k_grid]
#             + base_evals
#         ),
#     ],
#     [
#         "launch_configs/fiction/data_configs/train_val_splits_50pct/tuolumne_doc_split_train_val.json",
#         ",".join(
#             [
#                 f"style_strat_doc_split_fictions_{spl}_ds_valct1_styleNone_seed1234_mcq_topk{k}"
#                 for spl, k in split_k_grid
#             ]
#             + base_evals
#         ),
#     ],
#     [
#         "launch_configs/fiction/data_configs/train_val_splits_50pct/tuolumne_style_split_loo_blog_train_val.json",
#         ",".join(
#             [
#                 f"style_strat_doc_split_fictions_{spl}_ds_valctNone_styleblog_seed1234_mcq_topk{k}"
#                 for spl, k in split_k_grid
#             ]
#             + base_evals
#         ),
#     ],
#     [
#         "launch_configs/fiction/data_configs/train_val_splits_50pct/tuolumne_style_split_loo_news_train_val.json",
#         ",".join(
#             [
#                 f"style_strat_doc_split_fictions_{spl}_ds_valctNone_stylenews_seed1234_mcq_topk{k}"
#                 for spl, k in split_k_grid
#             ]
#             + base_evals
#         ),
#     ],
# ]
# exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))


seed_hparam = [
    1234,
    # 1337,
    # 1823,
    # 4321,
    # 9669,
]
final_exp_list = list(chain(*[[exp + [hp] for hp in seed_hparam] for exp in exp_list]))

if JOB_LIMIT is not None:
    final_exp_list = final_exp_list[:JOB_LIMIT]

for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        nodes,
        mbsz,
        wbsz,
        block_size,
        grad_ckpt,
        train_cfg,
        data_cfg,
        lm_eval_tasks,
        seed,
    ) = exp

    train_config_str = train_cfg.split("/")[-1].strip(".json").replace("tuolumne_", "").replace("_", "-")
    data_config_str = data_cfg.split("/")[-1].strip(".json").replace("tuolumne_", "").replace("_", "-")

    run_name = f"{BASE_RUN_NAME}_{nodes}N_mb{mbsz}-wb{wbsz}_{train_config_str}_{data_config_str}"
    # seed not in name so that results can be aggregated in wandb for each run name
    # and version out front for sorting

    if DO_TRAIN:
        command = f"""\
python {LAUNCHER_FILEPATH} \
--output_dir={BASE_OUT_DIR}/{EXP_VERSION}_{BASE_RUN_NAME}_seed{seed} \
--rccl_installdir={RCCL_INSTALL_DIR} \
--rccl_mode={RCCL_MODE} \
--qos={QOS} \
--rocm_version={ROCM_VERSION} \
--run_name={run_name} \
--nodes={nodes} \
--minutes={TIME_LIMIT} \
--custom_invocation='python -u train.py --config={train_cfg} --micro_batch_size={mbsz} --world_batch_size={wbsz} --block_size={block_size} --data_config={data_cfg} --max_steps={MAX_STEPS} --eval_step_interval={EVAL_INTERVAL} --eval_iters={EVAL_ITERS} --save_step_interval={SAVE_INTERVAL} --save_first_step=True --gradient_checkpointing={grad_ckpt} --seed={seed} --wandb_offline={WANDB_OFFLINE} --wandb_tags={WANDB_TAG_STRING} --logger_project={WANDB_PROJECT_NAME}' \
{f'--dryrun' if WRITE_ONLY else ''}\
"""

    custom_invocation = ""

    parent_dir = f"{BASE_OUT_DIR}/{EXP_VERSION}_{BASE_RUN_NAME}_seed{seed}/{run_name}"
    checkpoint_dir = f"{parent_dir}/checkpoints-FSDPStrategy"

    if DO_CONVERT:

        if not os.path.exists(parent_dir):
            continue

        ckpt_pattern = os.path.join(checkpoint_dir, f"*.pth")
        ckpts_found = glob.glob(ckpt_pattern)

        step_numbers = []
        for ckpt in ckpts_found:
            match = re.search(r"step-(\d+)", ckpt)
            step_number = match.group(1) if match else None
            if step_number is not None:
                step_numbers.append(step_number)

        step_numbers = sorted(step_numbers)
        # print(step_numbers)

        if EVAL_STEPS is not None:
            step_numbers = [step for step in step_numbers if int(step) in EVAL_STEPS]

        with open(f"{parent_dir}/run_config.json", "r") as fp:
            run_config = json.load(fp)
        tokenizer_dir = run_config["tokenizer_path"]

        for step_number in step_numbers:
            if os.path.exists(f"{parent_dir}/hf_checkpoints/hf_checkpoint_step-{step_number}-{run_name}"):
                # We assume this one is done if it got moved successfully, edge case is unlikely
                continue

            if custom_invocation != "":
                custom_invocation += "\n\n"
            custom_invocation += f"""\
rm -rf {parent_dir}/lit_checkpoint_step-{step_number}-{run_name}
rm -rf {parent_dir}/hf_checkpoint_step-{step_number}-{run_name}

python scripts/convert_checkpoint_to_hf.py \
--parent_dir {parent_dir} \
--checkpoint_file {checkpoint_dir}/step-{step_number}-{run_name}.pth \
--tokenizer_dir {tokenizer_dir} \
--model_name step-{step_number}-{run_name} \
--push_to_hub {PUSH_TO_HUB}

mkdir -p {parent_dir}/hf_checkpoints
mv {parent_dir}/hf_checkpoint_step-{step_number}-{run_name} {parent_dir}/hf_checkpoints/hf_checkpoint_step-{step_number}-{run_name}
rm -rf {parent_dir}/lit_checkpoint_step-{step_number}-{run_name}
"""

    if DO_EVAL:

        ckpt_pattern = os.path.join(checkpoint_dir, f"*.pth")
        ckpts_found = glob.glob(ckpt_pattern)

        step_numbers = []
        for ckpt in ckpts_found:
            match = re.search(r"step-(\d+)", ckpt)
            step_number = match.group(1) if match else None
            if step_number is not None:
                step_numbers.append(step_number)

        step_numbers = sorted(step_numbers)
        # print(step_numbers)

        if EVAL_STEPS is not None:
            step_numbers = [step for step in step_numbers if int(step) in EVAL_STEPS]

        for step_number in step_numbers:
            if custom_invocation != "":
                custom_invocation += "\n\n"
            custom_invocation += f"""\
lm_eval --model hf \
--model_args pretrained={parent_dir}/hf_checkpoints/hf_checkpoint_step-{step_number}-{run_name} \
--tasks {lm_eval_tasks} \
--device cuda:0 \
--batch_size {LM_EVAL_BATCH_SIZE} \
--output_path {parent_dir}/lm_eval_results \
--wandb_args project={WANDB_PROJECT_NAME},dir={parent_dir}/lm_eval_results,name={run_name},step={int(step_number)},tags={EXP_VERSION},notes={EXP_VERSION}
"""
    # --log_samples \

    if DO_CONVERT or DO_EVAL:
        gpus_per_node = f" --gpus_per_node={EVAL_GPUS_PER_NODE}" if EVAL_GPUS_PER_NODE is not None else ""

        command = f"""\
python {LAUNCHER_FILEPATH} \
--output_dir={BASE_OUT_DIR}/{EXP_VERSION}_{BASE_RUN_NAME}_seed{seed} \
--rccl_installdir={RCCL_INSTALL_DIR} \
--rccl_mode={RCCL_MODE} \
--qos={QOS} \
--rocm_version={ROCM_VERSION} \
--run_name={run_name} \
--nodes={EVAL_NODE_COUNT if EVAL_NODE_COUNT is not None else nodes}{gpus_per_node} \
--minutes={TIME_LIMIT} \
--custom_invocation='{custom_invocation}' \
--pass_run_name=False \
{f'--dryrun' if WRITE_ONLY else ''}\
"""

        if custom_invocation == "":
            command = f'echo "Nothing to run for {run_name}"'

    total_launches += 1
    if not DRY_RUN:
        os.system(command)
    else:
        print(run_name)

print(f"Total launches: {total_launches}")
