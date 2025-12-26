#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tctrl

cd ./Megatron-LM
# Megetron-LM fc
# export E2_LOG=1
export E2_LOG_X=1
export LOG_FILE="../logs/e2.csv"
export FIX_TIDS_PREFIX="../logs/bc_p5_nodes_rank"

# : > $LOG_FILE
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 6.0

# : > $LOG_FILE
# echo "mem_budge,0.7" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 5.2

# : > $LOG_FILE
# echo "mem_budge,0.6" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 4.5

: > $LOG_FILE
echo "mem_budge,0.5" >> $LOG_FILE
bash ./examples/pretrain_llama_distributed.sh 3.7

# : > $LOG_FILE
# echo "mem_budge,0.4" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.0

# : > $LOG_FILE
# echo "mem_budge,0.3" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 2.2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# export FIX_TIDS_PREFIX="../logs/cc_p5_nodes_rank"

# : > $LOG_FILE
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 6.0

# : > $LOG_FILE
# echo "mem_budge,0.7" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 5.2

# : > $LOG_FILE
# echo "mem_budge,0.6" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 4.5

# : > $LOG_FILE
# echo "mem_budge,0.5" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.7

# : > $LOG_FILE
# echo "mem_budge,0.4" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.0

# : > $LOG_FILE
# echo "mem_budge,0.3" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 2.2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# export FIX_TIDS_PREFIX="../logs/dc_p5_nodes_rank"

# : > $LOG_FILE
# echo "mem_budge,0.8" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 6.0

# : > $LOG_FILE
# echo "mem_budge,0.7" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 5.2

# : > $LOG_FILE
# echo "mem_budge,0.6" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 4.5

# : > $LOG_FILE
# echo "mem_budge,0.5" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.7


# : > $LOG_FILE
# echo "mem_budge,0.4" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 3.0

# : > $LOG_FILE
# echo "mem_budge,0.3" >> $LOG_FILE
# bash ./examples/pretrain_llama_distributed.sh 2.2

unset E2_LOG
unset E2_LOG_X
unset LOG_FILE
unset FIX_TIDS_PREFIX