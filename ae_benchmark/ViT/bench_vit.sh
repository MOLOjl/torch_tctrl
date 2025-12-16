mode=$1
if [ "$mode" == "ours" ]; then
    export DTR_ENABLE=1
    export COST_FIRST_EVICT=0
    # export DAG_LOCK_ENABLE=1
else
    export DTR_ENABLE=1
    export COST_FIRST_EVICT=1
    export ORIG_DTR=1
fi


# for gmlake
# export GMLAKE_INFO=1
# export fragment_limit=33554432

# 定义 mem_budgets 数组
mem_budgets=(8 3.02 2.64 2.27 1.89 1.51 1.13 0.75)
# 循环遍历 mem_budgets 数组
for mem_budget in "${mem_budgets[@]}"
do
    export MEM_BUDGET=$mem_budget
    echo "Running with MEM_BUDGET=$MEM_BUDGET"
    python vit_torch.py
done
