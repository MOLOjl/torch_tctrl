#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1


DATA_PATH=/home/sunzhenbo/dataset/nlp/enwik8_text_document

export GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --initial-loss-scale 524288 \
    --sequence-parallel \
    --timing-log-level 2 \
    --timing-log-option all \
    --use-distributed-optimizer \
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

RECOMPUTE_ARGS="
    --profile-memory \
    --recompute-config recompute_config/even_part/gpt_32layer_1dp_2mp_4pp_32ms_29mem.json
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 5 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"
PROFILE_ARGS="
    --profile \
    --profile-step-start 20 \
    --profile-step-end 22 \
    --profile-ranks 2
"

#nsys profile -s none -t nvtx,cuda -o nsys_rank_${RANK} --cuda-memory-usage true --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \

~/.venvs/torch2/bin/python3 pretrain_gpt.py \
    $GPT_ARGS \
    $RECOMPUTE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
