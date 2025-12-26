#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tctrl

cd ./Megatron-LM
# Megetron-LM fc
export E2_LOG=1
export LOG_FILE="../logs/e2.csv"

: > $LOG_FILE
echo "mem_budge,0.8" >> $LOG_FILE
bash ./examples/pretrain_llama_distributed.sh 6.0
python ../dump_node.py $LOG_FILE

# : > $LOG_FILE
# echo "mem_budge,0.7" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 5.2
# python ../dump_node.py $LOG_FILE

# : > $LOG_FILE
# echo "mem_budge,0.6" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 4.5
# python ../dump_node.py $LOG_FILE

# : > $LOG_FILE
# echo "mem_budge,0.5" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.7
# python ../dump_node.py $LOG_FILE

# : > $LOG_FILE
# echo "mem_budge,0.4" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.0
# python ../dump_node.py $LOG_FILE

# : > $LOG_FILE
# echo "mem_budge,0.3" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 2.2
# python ../dump_node.py $LOG_FILE

unset E2_LOG
unset LOG_FILE