#!/bin/bash

EXPECTED_ARGS=2

# 检查参数个数是否符合期望
if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: $0 <accelerate_config.yaml> <devices like 0,1,2,3>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$2
export MULTI_GPU=1
export LORA_ENABLE=1


export DTR_ENABLE=1  # 代码历史原因，直接注释是禁用
export COST_FIRST_EVICT=0   # 1是cost first 0是mem first，前者快碎片率高，后者慢碎片率低, gmlake请使用1
export DAG_LOCK_ENABLE=1
# export DAG_GRAPH_CONSTRAINT_SIZE=10000000
# export DAG_UPDATE_STABLE_STRIDE=25

export RESIDUAL_DEGREE=4
export CHAIN_LOCK_STRIDE=1
export WANDB_MODE=offline

MODEL_PATH=/data/wangzehua/model_space/Llama-2-7b-hf
DATASET_PATH=/data/wangzehua/dataset/stack-exchange-paired

# 定义一个包含mem_budget数值的列表
# mem_budgets=(8 5.39 4.72 4.04 3.37 2.7 2.02)  # Llama2-7B org
mem_budgets=(8)  # Llama2-7B org

# 遍历mem_budgets数组
for mem_budget in "${mem_budgets[@]}"
do
    # 使用 'wait' 确保命令串行执行
    (accelerate launch --config_file $1 sft_Llama2.py \
    --model_name=$MODEL_PATH \
    --dataset_name=$DATASET_PATH \
    --training_args.per_device_train_batch_size=4 \
    --training_args.per_device_eval_batch_size=1  \
    --training_args.gradient_accumulation_steps=2 \
    --training_args.max_steps=50     \
    --training_args.warmup_steps=10    \
    --training_args.mem_budget=$mem_budget) &
    pid=$!
    wait $pid

    # 输出当前的内存预算，用于调试确认
    echo "Running with mem_budget: $mem_budget finished"
done

