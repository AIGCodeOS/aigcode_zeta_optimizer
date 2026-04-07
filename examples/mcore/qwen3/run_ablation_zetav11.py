import os
import subprocess
import sys
import argparse
import time
import socket
import random
from contextlib import closing
from datetime import datetime

# ==============================================================================
# ZetaV11 Ablation Experiment Launcher
# ==============================================================================
# This script launches ablation experiments for the ZetaV11 optimizer.
# It iterates over defined ablation groups, constructs the training command,
# and executes it (sequentially or in parallel).
#
# Usage:
#   python examples/mcore/qwen3/run_ablation_zetav11.py --parallel 4
# ==============================================================================

# 1. Define Ablation Groups
# Keys are experiment names, Values are lists of arguments to append to the base command.
ABLATION_GROUPS = {
    # "baseline": [],  # All defaults (True) - Already run
    "no_tunneling": ["--no-zeta-use-tunneling"],
    "no_plasma": ["--no-zeta-use-plasma-confinement"],
    "no_soft_ns": ["--no-zeta-use-soft-ns"],
    "no_cayley": ["--no-zeta-use-cayley"],
}

def find_free_port():
    """
    Finds a free port on localhost.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_base_command(args, experiment_name, master_port):
    """
    Constructs the base torchrun command for Qwen3 training with ZetaV11.
    """
    
    # Distributed settings
    # TP=1, PP=4 => Need 4 GPUs per job
    NPUS_PER_NODE = 4
    MASTER_ADDR = "127.0.0.1" # Use explicit IP to avoid localhost resolution issues
    MASTER_PORT = master_port
    NNODES = 1
    NODE_RANK = 0
    
    DISTRIBUTED_ARGS = [
        "--nproc_per_node", str(NPUS_PER_NODE),
        "--nnodes", str(NNODES),
        "--node_rank", str(NODE_RANK),
        "--master_addr", MASTER_ADDR,
        "--master_port", str(MASTER_PORT),
    ]

    # Model args
    TP = 1
    PP = 4
    MBS = 4
    GBS = 256
    SEQ_LENGTH = 4096
    TRAIN_ITERS = 20000

    MODEL_PARALLEL_ARGS = [
        "--tensor-model-parallel-size", str(TP),
        "--pipeline-model-parallel-size", str(PP),
    ]

    TRAIN_ARGS = [
        "--micro-batch-size", str(MBS),
        "--global-batch-size", str(GBS),
        "--lr", "3e-4",
        "--min-lr", "1.25e-7",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--weight-decay", "2e-1",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--clip-grad", "1.0",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--initial-loss-scale", "4096",
        "--seed", "42",
        "--bf16",
        "--train-iters", str(TRAIN_ITERS),
        "--seq-length", str(SEQ_LENGTH),
        "--no-shared-storage",
    ]

    # Critical: Use ZetaV11
    OPTIMIZE_ARGS = [
        "--optimizer-selection", "zetav11", # ENABLE ZETAV11
        "--use-flash-attn",
        "--use-fused-rotary-pos-emb",
        "--use-rotary-position-embeddings",
        "--use-fused-swiglu",
        "--use-fused-rmsnorm",
        "--no-masked-softmax-fusion",
    ]

    GPT_ARGS = [
        "--use-mcore-models",
        "--sequence-parallel",
        "--spec", "mindspeed_llm.tasks.models.spec.qwen3_spec", "layer_spec",
        "--kv-channels", "128",
        "--qk-layernorm",
        "--norm-topk-prob",
        "--num-layers", "28",
        "--hidden-size", "1024",
        "--num-attention-heads", "16",
        "--ffn-hidden-size", "3072",
        "--max-position-embeddings", "32768",
        "--make-vocab-size-divisible-by", "1",
        "--padded-vocab-size", "151936",
        "--rotary-base", "1000000",
        "--disable-bias-linear",
        "--swiglu",
        "--tokenizer-type", "PretrainedFromHF",
        "--tokenizer-name-or-path", args.tokenizer_path,
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--norm-epsilon", "1e-6",
        "--no-gradient-accumulation-fusion",
        "--attention-softmax-in-fp32",
        "--exit-on-missing-checkpoint",
        "--group-query-attention",
        "--num-query-groups", "8",
        "--init-method-std", "0.01",
        "--no-load-optim",
        "--no-load-rng",
        "--seed", "42",
        "--bf16",
    ]

    DATA_ARGS = [
        "--data-path"] + args.data_path.split() + [
        "--split", "100,0,0",
    ]

    # Unique output dir for each experiment
    exp_save_dir = os.path.join(args.output_dir, experiment_name)
    wandb_dir = os.path.join(args.output_dir, "wandb", experiment_name)
    tensorboard_dir = os.path.join(args.output_dir, "tensorboard", experiment_name)
    
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs(wandb_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    OUTPUT_ARGS = [
        "--log-interval", "1",
        "--save-interval", str(TRAIN_ITERS),
        "--eval-interval", str(TRAIN_ITERS),
        "--eval-iters", "0",
        "--save", exp_save_dir,
        "--log-throughput",
        "--tensorboard-dir", tensorboard_dir,
        "--log-timers-to-tensorboard",
        "--use-wandb",
        "--wandb-project", "qwen3-0.6b",
        "--wandb-exp-name", f"zetav11_ablation_{experiment_name}",
        "--wandb-save-dir", wandb_dir,
    ]
    
    # Load from ckpt if provided (optional)
    if args.load_dir:
        OUTPUT_ARGS.extend(["--load", args.load_dir])

    # Construct full command
    # Use absolute path for pretrain_gpt.py to ensure it works from any directory
    pretrain_script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "pretrain_gpt.py")
    
    cmd = ["torchrun"] + DISTRIBUTED_ARGS + [pretrain_script] + \
          GPT_ARGS + DATA_ARGS + OUTPUT_ARGS + OPTIMIZE_ARGS + \
          TRAIN_ARGS + MODEL_PARALLEL_ARGS + \
          ["--distributed-backend", "nccl"]
          
    return cmd, exp_save_dir

def get_default_data_path():
    DATASET_BASE_DIRS = [
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_1_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_2_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_3_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_4_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_5_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_6_of_10",
        "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_7_of_10",
    ]
    DATA_RANGE = [
        (0, 278), (0, 278), (0, 278), (0, 278), (0, 278), (0, 278), (0, 278), (0, 278)
    ]
    data_path_list = []
    for idx, base_dir in enumerate(DATASET_BASE_DIRS):
        start, end = DATA_RANGE[idx]
        for i in range(start, end + 1):
            part_name = f"shard_{i:08d}_processed_text_document"
            data_path_list.append(os.path.join(base_dir, part_name))
    return " ".join(data_path_list)

def main():
    parser = argparse.ArgumentParser(description="Run ZetaV11 Ablation Experiments")
    parser.add_argument("--data-path", type=str, default=get_default_data_path(), help="Path to training data")
    parser.add_argument("--tokenizer-path", type=str, default="/sharedata/data/models/Qwen3-0.6B-Base", help="Path to tokenizer")
    parser.add_argument("--output-dir", type=str, default="/sharedata/ckw/result/ablation_zetav11", help="Root directory for outputs")
    parser.add_argument("--load-dir", type=str, help="Optional checkpoint load directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--experiment", type=str, choices=list(ABLATION_GROUPS.keys()) + ["all"], default="all", help="Specific experiment to run")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent experiments to run (default: 1)")
    
    args = parser.parse_args()

    experiments_to_run = [args.experiment] if args.experiment != "all" else list(ABLATION_GROUPS.keys())
    
    # Parallel execution management
    running_procs = [] # List of (process, experiment_name, slot_id)
    available_slots = list(range(args.parallel))
    
    # We assume each job needs 4 GPUs (TP=1, PP=4)
    GPUS_PER_JOB = 4
    
    # Base port for distributed training (ensure enough spacing)
    BASE_PORT = 6000
    print(f"Starting ablation experiments. Parallel jobs: {args.parallel}. GPUs per job: {GPUS_PER_JOB}")
    print(f"Using Base Port: {BASE_PORT} (incremented by 100 for each slot)")

    for exp_name in experiments_to_run:
        # Wait for a slot if not dry run
        if not args.dry_run:
            while not available_slots:
                # Check for finished processes
                for p, name, slot in running_procs[:]:
                    if p.poll() is not None: # Finished
                        running_procs.remove((p, name, slot))
                        available_slots.append(slot)
                        print(f"Experiment {name} finished with return code {p.returncode}.")
                
                if not available_slots:
                    time.sleep(5)
            
            slot = available_slots.pop(0)
        else:
            # For dry run, just assign a dummy slot to show intention
            slot = experiments_to_run.index(exp_name) % args.parallel

        # Assign resources
        # Use deterministic stride from random base to avoid race conditions
        # Increase spacing to 100 to avoid port conflicts with HCCL/torchrun
        master_port = BASE_PORT + (slot * 100)
        
        gpu_start = slot * GPUS_PER_JOB
        gpu_end = gpu_start + GPUS_PER_JOB
        gpus_list = [str(i) for i in range(gpu_start, gpu_end)]
        gpus_str = ",".join(gpus_list)
        
        print(f"\n{'='*80}")
        print(f"Preparing Experiment: {exp_name} (Slot {slot}, GPUs: {gpus_str}, Port: {master_port})")
        print(f"{'='*80}")

        ablation_args = ABLATION_GROUPS[exp_name]
        cmd, save_dir = get_base_command(args, exp_name, master_port=master_port)
        
        # Append ablation specific arguments
        cmd.extend(ablation_args)
        
        # Log file
        log_file = os.path.join(save_dir, "train.log")
        
        print(f"Command:\n{' '.join(cmd)}")
        print(f"Log file: {log_file}")
        
        if not args.dry_run:
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = gpus_str
            # Ensure clean separation
            env["OMP_NUM_THREADS"] = "1"
            
            # Start process
            print(f"Launching {exp_name} on GPUs {gpus_str}...")
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            running_procs.append((proc, exp_name, slot))
            
            # Add delay to avoid thundering herd on HCCL initialization
            print(f"Waiting 30s before launching next job to stabilize HCCL init...")
            time.sleep(30)
            
    # Wait for remaining processes
    if not args.dry_run:
        print("\nWaiting for remaining experiments to complete...")
        while running_procs:
            for p, name, slot in running_procs[:]:
                if p.poll() is not None:
                    running_procs.remove((p, name, slot))
                    print(f"Experiment {name} finished with return code {p.returncode}.")
            time.sleep(5)
            
    print("All experiments completed.")

if __name__ == "__main__":
    main()
