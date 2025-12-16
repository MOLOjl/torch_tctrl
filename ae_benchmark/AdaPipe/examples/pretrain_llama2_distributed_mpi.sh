#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

DATA_PATH=/mnt/share/dataset/nlp/enwik8_text_document

TP=$1
PP=$2
gbs=$3
seq_len=$4
mem=$5
prof=$6
log_dir=$7
layer_num=$8

iter_num=20

export GPUS_PER_NODE=8
export MASTER_ADDR=node006
#export MASTER_ADDR=`echo $MLP_MPI_HOSTS | awk '{split($1, arr, ":"); print arr[1]}'`
export MASTER_PORT=6000
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $layer_num \
    --hidden-size 8192 \
    --swiglu \
    --num-attention-heads 64 \
    --ffn-hidden-size 28672 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length $seq_len \
    --max-position-embeddings $seq_len \
    --micro-batch-size 1 \
    --global-batch-size $gbs \
    --lr 1.0e-5 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.0 \
    --clip-grad 1.0 \
    --initial-loss-scale 262144 \
    --sequence-parallel \
    --use-flash-attn \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --vocab-size 32000 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --use-distributed-optimizer \
    --log-dir $log_dir \
    --fp16
"

FULL_RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1
"

SELECTIVE_RECOMPUTE_ARGS="
    --recompute-granularity selective
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0
"

#nsys profile -s none -t nvtx,cuda -o nsys_rank_${RANK} --cuda-memory-usage true --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \

if [ $prof -eq 1 ]; then
    echo "profile recompute"
    PROFILE_ARGS="
    --profile-recompute \
    --train-iters 10 \
    --recompute-config recompute_config/llama2/baseline/${PP}pp.json
    "
elif [ $prof -eq 2 ]; then
    echo "evenpart"
    PROFILE_ARGS="
    --recompute-config recompute_config/llama2/evenpart/gpt_${seq_len}seq_${TP}mp_${PP}pp_${mem}mem.json \
    --profile-memory \
    --train-iters $iter_num
    "
else
    echo "adapipe"
    PROFILE_ARGS="
    --recompute-config recompute_config/llama2/adapipe/gpt_${seq_len}seq_${TP}mp_${PP}pp_${gbs}gbs_${mem}mem.json \
    --profile-memory \
    --train-iters $iter_num
    "
fi

#nsys profile -s none -t nvtx,cuda -o nsys_result/rank_${RANK} --cuda-memory-usage true --capture-range=cudaProfilerApi --capture-range-end=stop python3 pretrain_gpt.py \
#    --profile \
#    --profile-step-start 0 \
#    --profile-step-end 1 \

python3 pretrain_gpt.py \
    $GPT_ARGS \
    $PROFILE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
