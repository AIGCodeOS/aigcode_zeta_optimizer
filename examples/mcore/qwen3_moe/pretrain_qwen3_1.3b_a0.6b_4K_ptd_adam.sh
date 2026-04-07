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
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
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
MASTER_PORT=${USER_MASTER_PORT:-6000}
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

############################################
# 训练超参
############################################
LR=${USER_LR:-3e-4}
OPTIMIZER="adam" # 默认使用 adam

# 模型规模估算 (1.3B Total / 0.6B Active):
# Total = Embed(310M) + L*(Attn(6.3M) + E*FFN(3*H*H_moe))
# Active = Embed(310M) + L*(Attn(6.3M) + K*FFN(3*H*H_moe))
# 当 L=24, H=1024, E=16, K=4, H_moe=704 时:
# Total ≈ 310M + 24*(6.3M + 16*2.16M) ≈ 310M + 981M ≈ 1.29B
# Active ≈ 310M + 24*(6.3M + 4*2.16M) ≈ 310M + 358M ≈ 0.67B

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

TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
EXP_NAME="qwen3_moe_1.3b_4k_${OPTIMIZER}_lr${LR}"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
CKPT_LOAD_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="/sharedata/ckw/tensorboard/${EXP_NAME}"
WANDB_DIR="/sharedata/ckw/wandb"
WANDB_PROJECT="qwen3-moe-1.3b"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p logs

############################################
# 并行配置
############################################
TP=1
PP=1
EP=1
CP=1
SEQ_LENGTH=4096
MBS=1
GBS=256
TRAIN_ITERS=20000
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 16 \
    --moe-router-topk 4 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 704 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001 \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --gemm-gradient-accumulation-fusion \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr 1.0e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-iters 500 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 24 \
    --hidden-size 1024 \
    --ffn-hidden-size 1024 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --optimizer ${OPTIMIZER}
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
    --no-load-optim \
    --no-load-rng \
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
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/${EXP_NAME}.log