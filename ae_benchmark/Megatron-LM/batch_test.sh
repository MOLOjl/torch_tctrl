#!/bin/bash

# Define memory budgets to test
MEM_BUDGETS=(4.6 4.2 3.2 2.8 2.02)
ratios=(0.8 0.7 0.6 0.5 0.4)

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for i in "${!MEM_BUDGETS[@]}"; do
    budget=${MEM_BUDGETS[$i]}
    ratio=${ratios[$i]}
    echo "Running with MEM_BUDGET=$budget, RATIO=$ratio"
    LOG_FILE="pretrain_ours_${budget}_ratio${ratio}.log"
    
    # Call the original script with the current budget and log output
    ./pretrain_gpt_distributed.sh "$budget" > "$LOG_FILE" 2>&1
    
    echo "Completed run with MEM_BUDGET=$budget, RATIO=$ratio. Output saved to $LOG_FILE"
done

echo "All memory budget tests completed."