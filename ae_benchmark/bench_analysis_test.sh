#!/bin/bash

# 统计不同mem_budget下DTR和Megatron-LM的执行时间,重物化次数,平均递归深度,内存碎片率, 

cd ./Megatron-LM
export LOG_FILE="../logs/e1.csv"
bash ./examples/pretrain_gpt_distributed.sh 6.4

conda activate pyf_dtr
export LOG_FILE="../logs/e1_1.csv" 
bash ./examples/pretrain_gpt_distributed.sh 6.4
