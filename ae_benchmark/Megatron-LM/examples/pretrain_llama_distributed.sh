#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
######################################
# Change the below configurations here
DATASET=/data/wangzehua/dataset/oscar-en-10k/llama/oscar-en-10k-meg-llama_text_document
TOKENIZER_PATH=/data/wangzehua/model_space/Llama-2-13b-hf/tokenizer.model # offical llama tokenizer.model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7 # 4,5,6,7
export RECORD_MEM_SNAPSHOT=1
export SNAP_FILE_NAME="pretrain_llama_7B_fdtr_pp4_b36_distopt"

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


TP=1
PP=4
# ZERO_STAGE=0

# NUM_KV_HEADS=4 # llama2 70B uses GQA
SEQ_LENGTH=4096 # 2048
MODEL_SIZE="7"

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEADS=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEADS=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEADS=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEADS=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

MICRO_BATCH_SIZE=1      # 4
GLOBAL_BATCH_SIZE=32   # e.g. llama: 4M tokens
MAX_ITERS=20             # 250000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR_WARMUP_STEPS=1

USE_MEGATRON_LM_RC=0        # 是否启用Megatron-LM的重计算 1-selective 2-full

export DTR_ENABLE=1
export MEM_BUDGET=3.4
export RESIDUAL_DEGREE=4
export COST_FIRST_EVICT=0
export CHAIN_LENGTH_LOCK_THRESHOLD=4
export CHAIN_LOCK_STRIDE=2

LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters $MAX_ITERS \
    --lr $LR \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr $MIN_LR \
    --weight-decay $WEIGHT_DECAY \
    --clip-grad $GRAD_CLIP \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --bf16 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization RMSNorm \
    --no-rope-fusion \
    --disable-bias-linear
"
    # --use-distributed-optimizer

DATA_ARGS="
    --data-path $DATASET \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 2
"

# recomput selective | full
EXTRA_OPTIM_ARGS="
    --recompute-activations \
    --recompute-granularity selective
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
    # $EXTRA_OPTIM_ARGS \
        #\
    #    --num-key-value-heads $NUM_KV_HEADS #\
    #    $ds_args
        #    --no-query-key-layer-scaling \