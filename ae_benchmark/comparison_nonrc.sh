#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tctrl

cd ./Megatron-LM
# Megetron-LM fc
export E3_LOG=1

# T-Control (w/o TR)
: > $LOG_FILE
export DTR_ENABLE=0
sbatch ./examples/slurm/submit_gpt3_small_8.sh
sbatch ./examples/slurm/submit_llama3_small_8.sh
sbatch ./examples/slurm/submit_gpt3_large_128.sh
sbatch ./examples/slurm/submit_llama3_large_256.sh

# T-Control (with TR)
export DTR_ENABLE=1
sbatch ./examples/slurm/submit_gpt3_small_8.sh
sbatch ./examples/slurm/submit_llama3_small_8.sh
sbatch ./examples/slurm/submit_gpt3_large_128.sh
sbatch ./examples/slurm/submit_llama3_large_256.sh