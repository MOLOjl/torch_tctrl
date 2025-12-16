#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

DATA_PATH=/home/sunzhenbo/dataset/nlp/enwik8_text_document

TP=$1
PP=$2
gbs=$3
seq_len=$4
layer_num=$5
hidden_size=$6
head_num=$7
mem=$8
prof=$9
log_dir=${10}

iter_num=20

export GPUS_PER_NODE=8
export MASTER_ADDR=node002
export MASTER_PORT=6000
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $layer_num \
    --hidden-size $hidden_size \
    --num-attention-heads $head_num \
    --seq-length $seq_len \
    --max-position-embeddings $seq_len \
    --micro-batch-size 1 \
    --global-batch-size $gbs \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-iters 0 \
    --lr-decay-iters 430000 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --initial-loss-scale 131072 \
    --sequence-parallel \
    --use-distributed-optimizer \
    --untie-embeddings-and-output-weights \
    --log-dir $log_dir \
    --fp16
"

if [ $prof -eq 1 ]; then
    echo "profile recompute"
    PROFILE_ARGS="
    --profile-recompute \
    --train-iters 10 \
    --recompute-config recompute_config/gpt/baseline/${layer_num}ln_${PP}pp.json
    "
elif [ $prof -eq 2 ]; then
    echo "evenpart"
    PROFILE_ARGS="
    --recompute-config recompute_config/gpt/evenpart/gpt_${seq_len}seq_${hidden_size}hidden_${layer_num}ln_${TP}mp_${PP}pp_${mem}mem.json \
    --profile-memory \
    --train-iters $iter_num
    "
elif [ $prof -eq 3 ]; then
    echo "adapipe"
    PROFILE_ARGS="
    --recompute-config recompute_config/gpt/adapipe/gpt_${seq_len}seq_${hidden_size}hidden_${layer_num}ln_${TP}mp_${PP}pp_${gbs}gbs_${mem}mem.json \
    --profile-memory \
    --train-iters $iter_num
    "
else
    echo "convergence"
    PROFILE_ARGS="
    --recompute-config recompute_config/gpt/adapipe/gpt_${seq_len}seq_${hidden_size}hidden_${layer_num}ln_${TP}mp_${PP}pp_${gbs}gbs_${mem}mem.json \
    --train-iters 300
    "
fi


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

python3 pretrain_gpt.py \
    $GPT_ARGS \
    $PROFILE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
