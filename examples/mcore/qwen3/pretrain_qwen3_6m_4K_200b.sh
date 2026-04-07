#!/bin/bash

# Qwen3 6M with ZetaHS (Hodge-Soliton)
# 合并 zeta.py (Hodge分解) + zeta_soliton.py (孤子检测)

############################################
# 基础环境
############################################
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export PATH="/sharedata/shareenvs/qiuwu-optimizer-dev/bin:${PATH}"

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6042
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

############################################
# 训练超参
############################################
LR=$2
MIN_LR="1.25e-7"
OPTIMIZER=$1

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
HOST_NAME=`hostname`
OPT=$OPTIMIZER"_"$3
EXP_NAME="qwen3_6m_4k_"$OPT"_lr${LR}_200b_${HOST_NAME/-*/}"
CKPT_SAVE_DIR="/sharedata/qiuwu/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="./tensorboard/${EXP_NAME}"
WANDB_DIR="./wandb/${EXP_NAME}"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p logs

############################################
# 并行配置 (6M模型很小)
############################################
TP=1
PP=1

############################################
# Batch / Token 对齐
############################################
SEQ_LEN=4096
MBS=4
GBS=64
TRAIN_ITERS=${TRAIN_ITERS_OVERRIDE:-20000}

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
# 模型参数 (Qwen3 ~6M)
############################################
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    
    --num-layers 4 \
    --hidden-size 256 \
    --num-attention-heads 4 \
    --ffn-hidden-size 768 \
    
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    
    --kv-channels 64 \
    --qk-layernorm \
    --norm-topk-prob \
    
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    
    --normalization RMSNorm \
    --swiglu \
    --disable-bias-linear \
    
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 2 \
    
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.01 \
    
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --bf16
"

############################################
# 优化器参数
############################################
OPTIM_ARGS="
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --train-iters ${TRAIN_ITERS} \
    --weight-decay 2e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    
    --optimizer ${OPTIMIZER} \
    
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion
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
    --wandb-project qwen3-6m-hodge-soliton \
    --wandb-exp-name ${EXP_NAME} \
    --wandb-save-dir ${WANDB_DIR}
"

############################################
# 启动训练
############################################
pre=/sharedata/qiuwu/230b
dir=/sharedata/shareenvs/qiuwu-optimizer-dev/bin/

echo "Starting Zeta${OPTIMIZER}: ${EXP_NAME}"
#echo $DATA_ARGS

$dir/torchrun $DISTRIBUTED_ARGS $pre/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${OPTIM_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --seed 42 \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/${EXP_NAME}.log
