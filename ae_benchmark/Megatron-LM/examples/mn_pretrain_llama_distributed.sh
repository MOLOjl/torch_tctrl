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
# CHECKPOINT_PATH=/home/bingxing2/home/scx6078/wangzehua/workspace/models/output/gpt2
# model_path=/home/bingxing2/home/scx6078/wangzehua/workspace/models/gpt2-large
# VOCAB_FILE=$model_path/vocab.json
# MERGE_FILE=$model_path/merges.txt
DATA_PATH=/home/bingxing2/home/scx6078/wangzehua/workspace/datasets/oscar-en-10k/oscar-en-10k/oscar-en-10k-meg-llama_text_document
TOKENIZER_PATH=/home/bingxing2/home/scx6078/luoyuan-workspace/models/llama-2-13b-hf/tokenizer.model

# 主机地址
MASTER_ADDR="${host[1]}"
echo MASTER:\ $MASTER_ADDR
echo
MASTER_PORT=6000
NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export CUDA_DEVICE_MAX_CONNECTIONS=1 # Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1

export LOG_MEM_EVENTS=1
export LOG_MEM_PREFIX=/home/bingxing2/home/scx6078/luoyuan-workspace/Megatron-LM/mem_logs

# 超参
TP_SIZE=1
PP_SIZE=4
MB=1
GLOBAL_BATCH=128
MAX_ITERS=50 # 500000

USE_MEGATRON_LM_RC=0        # 是否启用Megatron-LM的重计算 0 - no | 1 - selective | 2 - full

export DTR_ENABLE=1
export MEM_BUDGET=3.4        # TODO: 每个都调整还是统一
export RESIDUAL_DEGREE=6
export CHAIN_LOCK_STRIDE=4
export COST_FIRST_EVICT=0  # 1是cost first 0是mem first，前者快碎片率高，后者慢碎片率低
# export LOG_CUDAAPI=1        # 记录累计的cuda api次数

# 模型配置
SEQ_LENGTH=4096 # 2048
MODEL_SIZE="7"

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=4096;  NUM_HEADS=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEADS=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEADS=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEADS=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

LR_WARMUP_STEPS=10

LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1

LLAMA_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NUM_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MB \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $MAX_ITERS \
    --lr $LR \
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

DATA_ARGS="
    --data-path $DATA_PATH \
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
    # export MEM_BUDGET=3
    # export RECORD_MEM_SNAPSHOT=0 # aarch64 not support
    # export SNAP_FILE_NAME="pretrain_gpt_350M_dtr_chain.pickle"

    # Change for multinode config
    NODE_RANK=$i

    DISTRIBUTED_ARGS="
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "
    echo $DISTRIBUTED_ARGS

    RUN="srun -N 1 --gres=gpu:$GPUS_PER_NODE -w ${host[$((i+1))]} \
        torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
            $LLAMA_ARGS \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            $EXTRA_OPTIM_ARGS \
            --distributed-backend nccl"
    if [ $i == $n ]; then
        $RUN >> ${SLURM_JOBID}_train_rank${i}_${host[$((i+1))]}.log 2>&1
    else
        $RUN >> ${SLURM_JOBID}_train_rank${i}_${host[$((i+1))]}.log 2>&1 &
    fi
done
wait

