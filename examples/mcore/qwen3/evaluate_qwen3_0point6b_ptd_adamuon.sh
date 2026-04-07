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
# 评测配置 (与训练脚本对齐)
############################################
LR=${USER_LR:-3e-4}
OPTIMIZER="adamuon"
EXP_NAME="qwen3_0point6b_4k_${OPTIMIZER}_lr${LR}"

# 路径配置
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
CHECKPOINT="/sharedata/ckw/ckpt/${EXP_NAME}"
DATA_PATH="/sharedata/ckw/data/mmlu/data/test/" # 请根据实际评测数据路径修改
TASK="mmlu"

TP=1
PP=4
EP=1
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
         --no-chat-template \
         --task-data-path ${DATA_PATH} \
         --task ${TASK} \
         --use-mcore-models \
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --expert-model-parallel-size ${EP} \
         --load ${CHECKPOINT} \
         --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
         --kv-channels 128 \
         --norm-topk-prob \
         --use-fused-rotary-pos-emb \
         --use-rotary-position-embeddings \
         --use-fused-swiglu \
         --use-fused-rmsnorm \
         --qk-layernorm \
         --num-layers 28 \
         --hidden-size 1024 \
         --use-rotary-position-embeddings \
         --num-attention-heads 16 \
         --ffn-hidden-size 3072 \
         --max-position-embeddings ${SEQ_LENGTH} \
         --seq-length ${SEQ_LENGTH} \
         --make-vocab-size-divisible-by 1 \
         --padded-vocab-size 151936 \
         --rotary-base 1000000 \
         --micro-batch-size 1 \
         --disable-bias-linear \
         --swiglu \
         --tokenizer-type PretrainedFromHF \
         --tokenizer-name-or-path ${TOKENIZER_PATH} \
         --normalization RMSNorm \
         --position-embedding-type rope \
         --norm-epsilon 1e-6 \
         --hidden-dropout 0 \
         --attention-dropout 0 \
         --max-new-tokens 2 \
         --no-gradient-accumulation-fusion \
         --attention-softmax-in-fp32 \
         --exit-on-missing-checkpoint \
         --no-masked-softmax-fusion \
         --group-query-attention \
         --num-query-groups 8 \
         --seed 42 \
         --bf16 \
         | tee logs/evaluate_${EXP_NAME}.log