#!/bin/bash

### 加载环境
module load compilers/gcc/12.2.0 compilers/cuda/11.8   cudnn/8.6.0.163_cuda11.x  nccl/2.11.4-1_cuda11.8
source activate dtb
export PYTORCH_BUILD_NUMBER=1
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTORCH_BUILD_VERSION="2.1.0a0+cu118"
export NCCL_ROOT=/home/bingxing2/apps/nccl/2.11.4-1_cuda11.8/build

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
# export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=22

### 获取每个节点的host
k=0
for i in `scontrol show hostnames`
do
    let k=k+1
    host[$k]=$i
    echo ${host[$k]}
done

### 统一参数 ###
CHECKPOINT_PATH=/home/bingxing2/home/scx6078/wangzehua/workspace/models/output/gpt2
model_path=/home/bingxing2/home/scx6078/wangzehua/workspace/models/gpt2-large
VOCAB_FILE=$model_path/vocab.json
MERGE_FILE=$model_path/merges.txt
DATA_PATH=/home/bingxing2/home/scx6078/wangzehua/workspace/datasets/oscar-en-10k/oscar-en-10k/oscar-en-10k-meg-GPT_text_document

# 主机地址
MASTER_ADDR="${host[1]}"
echo MASTER:\ $MASTER_ADDR
echo
MASTER_PORT=6000
NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1

MAX_ITERS=50 # 500000

USE_MEGATRON_LM_RC=0       # 是否启用Megatron-LM的重计算 0 - no | 1 - selective | 2 - full

# export DTR_ENABLE=1
# export MEM_BUDGET=3.2        # TODO: 每个都调整还是统一
export RESIDUAL_DEGREE=6
export CHAIN_LOCK_STRIDE=4
export COST_FIRST_EVICT=0  # 1是cost first 0是mem first，前者快碎片率高，后者慢碎片率低
# export E1_POOL_MAX=10485760   #  10 350M-67108864    
# export E2_POOL_MAX=37748736  # 36  350M-268435456
# export LOG_CUDAAPI=1        # 记录累计的cuda api次数
# export LOG_MEM_EVENTS=1        # 记录CUDA MEM事件
# export LOG_MEM_PREFIX=/home/bingxing2/home/scx6078/wangzehua/workspace/Megatron-LM/logs/mem_logs


# 模型配置 和 超参
GLOBAL_BATCH=128
if [[ $NNODES -eq 8 ]]; then
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    ATTENTION_HEADS=16
    MB=4
    TP_SIZE=1
    if [[ $DTR_ENABLE -eq 1 ]]; then
        PP_SIZE=1
    else
        PP_SIZE=2
    fi
else
    NUM_LAYERS=96
    HIDDEN_SIZE=12288
    ATTENTION_HEADS=96
    MB=1
    TP_SIZE=4
    PP_SIZE=32
fi

MAX_SEQ_LEN=2048

GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $ATTENTION_HEADS \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_SEQ_LEN \
    --micro-batch-size $MB \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.00015 \
    --train-iters $MAX_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
"
# --use-distributed-optimizer

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 2
"

EXTRA_OPTIM_ARGS=""
if [ $USE_MEGATRON_LM_RC -eq 1 ]; then
# recompute selective | full
EXTRA_OPTIM_ARGS="
    --recompute-activations \
    --recompute-granularity selective
"
fi
# recompute full
if [ $USE_MEGATRON_LM_RC -eq 2 ]; then
EXTRA_OPTIM_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1
"
fi


### 主节点运行 ###
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export MEM_BUDGET=3
# export RECORD_MEM_SNAPSHOT=0 # aarch64 not support
# export SNAP_FILE_NAME="pretrain_gpt_350M_dtr_chain.pickle"

# Change for multinode config
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 6
"

### 子节点运行 ###
n=$((NNODES-1))
for (( i=0; i<=n; i++ ))
do
    echo "Node-$i, host:${host[$((i+1))]}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Change for multinode config
    NODE_RANK=$i

    DISTRIBUTED_ARGS="
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    RUN="srun -N 1 --gres=gpu:$GPUS_PER_NODE -w ${host[$((i+1))]} \
        torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
            $GPT_ARGS \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            $EXTRA_OPTIM_ARGS \
            --distributed-backend nccl"
    if [ $i == $n ]; then
        $RUN >> ./logs/auto/${SLURM_JOBID}_train_rank${i}_${model_spec}_TP${TP_SIZE}_PP${PP_SIZE}_GB${GLOBAL_BATCH}_MB${MB}_${host[$((i+1))]}.log 2>&1
    else
        $RUN >> ./logs/auto/${SLURM_JOBID}_train_rank${i}_${model_spec}_TP${TP_SIZE}_PP${PP_SIZE}_GB${GLOBAL_BATCH}_MB${MB}_${host[$((i+1))]}.log 2>&1 &
    fi
done
wait

