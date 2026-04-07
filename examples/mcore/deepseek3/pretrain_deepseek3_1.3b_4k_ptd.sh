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
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=${USER_MASTER_PORT:-6000}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

############################################
# 训练超参
############################################
LR=${USER_LR:-1.0e-5}
OPTIMIZER="adam" # 默认使用 adam，可根据需要修改为 zeta 等

############################################
# 路径配置
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

TOKENIZER_PATH="/sharedata/ckw/models/deepseekv3"
EXP_NAME="deepseek3_1.3b_4k_${OPTIMIZER}_lr${LR}"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
CKPT_LOAD_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="/sharedata/ckw/tensorboard/${EXP_NAME}"
WANDB_DIR="/sharedata/ckw/wandb"
WANDB_PROJECT="deepseek3-1.3b"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p logs

############################################
# 并行配置
############################################
TP=1
PP=1
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=24
SEQ_LEN=4096
MBS=1
GBS=256

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MLA_ARGS="
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --q-lora-rank 1024 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
    --mla-mm-split \
    --mla-fa-without-pad \
"

MOE_ARGS="
    --moe-token-dispatcher-type alltoall \
    --moe-permute-fusion \
    --first-k-dense-replace 3 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 8 \
    --moe-router-topk 1 \
    --moe-ffn-hidden-size 1408 \
    --moe-shared-expert-intermediate-size 1408 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --moe-router-topk-scaling-factor 1.0 \
    --moe-aux-loss-coeff 0.0001 \
    --seq-aux \
    --norm-topk-prob \
    --disable-bias-linear
"

GPT_ARGS="
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 1024 \
    --ffn-hidden-size 1408 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr ${LR} \
    --train-iters 20000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.006 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 200 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --optimizer ${OPTIMIZER} \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-throughput \
    --use-wandb \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir ${WANDB_DIR}
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
    | tee logs/${EXP_NAME}.log
