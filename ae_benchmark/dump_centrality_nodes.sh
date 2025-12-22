#!/bin/bash

cd ./Megatron-LM
# Megetron-LM fc
export E2_LOG=1
export LOG_FILE="../logs/e2.csv"
echo "mem_budge,0.8" >> $LOG_FILE
bash ./examples/pretrain_llama_distributed.sh 6.0
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 5.2
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 4.5
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.7
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.0
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 2.2