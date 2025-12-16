#!/bin/bash


export DTR_ENABLE=1
export COST_FIRST_EVICT=0
export DAG_LOCK_ENABLE=1
# export DAG_GRAPH_CONSTRAINT_SIZE=10000
# export DAG_UPDATE_STABLE_STRIDE=5000

export BATCH_SIZE=10
export WARM_UP=1
export MAX_ITERS=10

mem_budgets=(8 6.16 5.39 4.62 3.85 3.08 2.31) 

for mem_budget in "${mem_budgets[@]}"
do
    python openfold_test.py $mem_budget
    echo "Running with mem_budget: $mem_budget finished"
done

