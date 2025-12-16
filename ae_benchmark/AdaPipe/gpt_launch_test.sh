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
layer_num=96
hidden_size=12288
head_num=96

tp=$TP
pp=$PP

baseline_strategy_dir=recompute_config/gpt/baseline
if [ ! -d $baseline_strategy_dir ]; then
    mkdir $baseline_strategy_dir
fi
evenpart_strategy_dir=recompute_config/gpt/evenpart
if [ ! -d $evenpart_strategy_dir ]; then
    mkdir $evenpart_strategy_dir
fi
adapipe_strategy_dir=recompute_config/gpt/adapipe
if [ ! -d $adapipe_strategy_dir ]; then
    mkdir $adapipe_strategy_dir
fi

baseline_strategy=recompute_config/gpt/baseline/${layer_num}ln_${PP}pp.json
python3 recompute_config/gpt/generate_baseline.py $PP $layer_num $baseline_strategy

log_dir=gpt_result/gbs${gbs}_seq${seq_len}_profile/log_tp${tp}_pp${pp}
result_name=gpt_result/gbs${gbs}_seq${seq_len}_profile/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP 32 $seq_len $layer_num $hidden_size $head_num $mem 1 $log_dir | tee $result_name
if [ $? -ne 0 ]; then
    mpirun -N 1 --hostfile $hostfile pkill -f pretrain
    exit
fi
python3 recompute_config/parser.py gpt $log_dir $world_size $TP $PP $seq_len $hidden_size

log_dir=gpt_result/gbs${gbs}_seq${seq_len}_evenpart/log_tp${tp}_pp${pp}
result_name=gpt_result/gbs${gbs}_seq${seq_len}_evenpart/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

PYTHONPATH=recompute_config python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $mem $seq_len $layer_num $hidden_size $head_num 1
mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP $gbs $seq_len $layer_num $hidden_size $head_num $mem 2 $log_dir | tee $result_name
mpirun -N 1 --hostfile $hostfile pkill -f pretrain

log_dir=gpt_result/gbs${gbs}_seq${seq_len}_adapipe/log_tp${tp}_pp${pp}
result_name=gpt_result/gbs${gbs}_seq${seq_len}_adapipe/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi
PYTHONPATH=recompute_config python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $mem $seq_len $layer_num $hidden_size $head_num 0
mpirun -n $world_size -N 8 --hostfile $hostfile sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP $gbs $seq_len $layer_num $hidden_size $head_num $mem 3 $log_dir | tee $result_name
mpirun -N 1 --hostfile $hostfile pkill -f pretrain
