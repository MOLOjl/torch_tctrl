#!/bin/bash

# 统计不同mem_budget下DTR和Megatron-LM的执行时间,重物化次数,平均递归深度,内存碎片率, 
export E1_LOG=1
cd ./Megatron-LM
# Megetron-LM fc
export LOG_FILE="../logs/e1.csv"
echo "mem_budge,0.8" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed_meLMfc.sh 6.0
echo "mem_budge,0.7" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed_meLMfc.sh 5.2
echo "mem_budge,0.6" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed_meLMfc.sh 4.5
echo "mem_budge,0.5" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed_meLMfc.sh 3.7
echo "mem_budge,0.4" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed_meLMfc.sh 3.0

# baseline DTR
conda activate pyf_dtr
export LOG_FILE="../logs/e1_1.csv"
echo "mem_budge,0.8" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed.sh 6.0
echo "mem_budge,0.7" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed.sh 5.2
echo "mem_budge,0.6" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed.sh 4.5
echo "mem_budge,0.5" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed.sh 3.7
echo "mem_budge,0.4" >> $LOG_FILE
bash ./examples/pretrain_gpt_distributed.sh 3.0