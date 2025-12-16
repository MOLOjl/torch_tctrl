#!/bin/bash

# profile
world_size=$1
TP=$2
PP=$3
gbs=$4
seq_len=$5
hostfile=$6

prof=2 # 1: profile, 2: evenpart, 3: adapipe
mem=70
layer_num=80
hidden_size=8192

tp=$TP
pp=$PP

baseline_strategy_dir=recompute_config/llama2/baseline
if [ ! -d $baseline_strategy_dir ]; then
    mkdir $baseline_strategy_dir
fi
evenpart_strategy_dir=recompute_config/llama2/evenpart
if [ ! -d $evenpart_strategy_dir ]; then
    mkdir $evenpart_strategy_dir
fi
adapipe_strategy_dir=recompute_config/llama2/adapipe
if [ ! -d $adapipe_strategy_dir ]; then
    mkdir $adapipe_strategy_dir
fi

baseline_strategy=recompute_config/llama2/baseline/${PP}pp.json
python3 recompute_config/llama2/generate_baseline.py $PP $layer_num $baseline_strategy

log_dir=llama2_result/gbs${gbs}_seq${seq_len}_profile/log_tp${tp}_pp${pp}
result_name=llama2_result/gbs${gbs}_seq${seq_len}_profile/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_llama2_distributed_mpi.sh $TP $PP 32 $seq_len $mem 1 $log_dir $layer_num | tee $result_name
if [ $? -ne 0 ]; then
    mpirun -N 1 --hostfile $hostfile pkill -f pretrain
    exit
fi
python3 recompute_config/parser.py llama2 $log_dir $world_size $TP $PP $seq_len $hidden_size

log_dir=llama2_result/gbs${gbs}_seq${seq_len}_evenpart/log_tp${tp}_pp${pp}
result_name=llama2_result/gbs${gbs}_seq${seq_len}_evenpart/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

PYTHONPATH=recompute_config python3 recompute_config/llama2/search.py $world_size $TP $PP $gbs $seq_len $mem $layer_num 1
mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_llama2_distributed_mpi.sh $TP $PP $gbs $seq_len $mem 2 $log_dir $layer_num | tee $result_name
mpirun -N 1 --hostfile $hostfile pkill -f pretrain

log_dir=llama2_result/gbs${gbs}_seq${seq_len}_adapipe/log_tp${tp}_pp${pp}
result_name=llama2_result/gbs${gbs}_seq${seq_len}_adapipe/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi
PYTHONPATH=recompute_config python3 recompute_config/llama2/search.py $world_size $TP $PP $gbs $seq_len $mem $layer_num 0
mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_llama2_distributed_mpi.sh $TP $PP $gbs $seq_len $mem 3 $log_dir $layer_num | tee $result_name
mpirun -N 1 --hostfile $hostfile pkill -f pretrain
