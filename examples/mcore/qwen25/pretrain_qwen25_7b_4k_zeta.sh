#!/bin/bash
set -e

# Positional overrides: 1=MASTER_PORT, 2=LR, 3=ASCEND_RT_VISIBLE_DEVICES
USER_MASTER_PORT=$1
USER_LR=$2
USER_VISIBLE=$3

# Allow overriding visible devices
if [[ -n "${USER_VISIBLE}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${USER_VISIBLE}
fi

############################################
# 基础环境
############################################
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

# Auto-detect NPUs unless NPUS_PER_NODE is preset
if [[ -z "${NPUS_PER_NODE:-}" || "${NPUS_PER_NODE}" == "auto" ]]; then
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra _asc_dev <<<"${ASCEND_RT_VISIBLE_DEVICES}"
        NPUS_PER_NODE=${#_asc_dev[@]}
    elif command -v npu-smi >/dev/null 2>&1; then
        NPUS_PER_NODE=$(npu-smi info 2>/dev/null | grep -c "Device ID")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NPUS_PER_NODE=8
    fi
fi

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=${USER_MASTER_PORT:-6003}
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

############################################
# 训练超参 (从 0.6B 脚本迁移)
############################################
LR=${USER_LR:-3e-4}
OPTIMIZER="zeta"

############################################
# 路径配置 (采用 0.6B 的训练集路径构造逻辑)
############################################
DATASET_BASE_DIRS=(
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_0_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_1_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_2_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_3_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_4_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_5_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_6_of_10"
    "/sharedata/data/indexed_data/dclm-baseline-1.0/global-shard_01_of_10/local-shard_7_of_10"
)
DATA_RANGE=(
    0 278
    0 278
    0 278
    0 278
    0 278
    0 278
    0 278
    0 278
)
DATA_PATH=""
for idx in "${!DATASET_BASE_DIRS[@]}"; do
    DATASET_BASE_DIR="${DATASET_BASE_DIRS[idx]}"
    start=${DATA_RANGE[idx * 2]}
    end=${DATA_RANGE[idx * 2 + 1]}
    for ((i = start; i <= end; i++)); do
        PART_NAME=$(printf "shard_%08d_processed_text_document" $i)
        DATA_PATH="$DATA_PATH $DATASET_BASE_DIR/$PART_NAME"
    done
done

TOKENIZER_PATH="/sharedata/ckw/model_from_hf/qwen2.5-0.5b-hf/"
CKPT_LOAD_DIR="/sharedata/ckw/ckpt/qwen2.5-7b-hf-zeta/"

EXP_NAME="qwen25_7b_4k_${OPTIMIZER}_lr${LR}"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="./tensorboard/${EXP_NAME}"
WANDB_DIR="./wandb/${EXP_NAME}"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p logs

############################################
# 并行配置 (7B 层数 28，PP=4 可整除)
############################################
TP=1
PP=7

############################################
# Batch / Token 对齐 (从 0.6B 脚本迁移)
############################################
SEQ_LEN=4096
MBS=1
GBS=64
TRAIN_ITERS=100000

############################################
# 分布式参数
############################################
DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

############################################
# 模型参数 (Qwen2.5 7B)
############################################
GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28  \
    --hidden-size 3584  \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.01 \
    --bf16
"

############################################
# 优化器参数 (zeta, 从 0.6B 脚本迁移)
############################################
OPTIM_ARGS="
    --lr ${LR} \
    --min-lr 1.25e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --train-iters ${TRAIN_ITERS} \
    --weight-decay 2e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    
    --optimizer ${OPTIMIZER} \
    
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32
"

############################################
# 数据参数
############################################
DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --no-shared-storage
"

############################################
# 日志 / 评测 / 保存
############################################
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-throughput \
    
    --use-wandb \
    --wandb-project qwen25-7b \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir ${WANDB_DIR}
"

############################################
# 启动训练
############################################
torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${GPT_ARGS} \
    ${OPTIM_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --seed 42 \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/${EXP_NAME}.log
