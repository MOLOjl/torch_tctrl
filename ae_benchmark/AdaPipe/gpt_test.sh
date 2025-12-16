#!/bin/bash

# profile
TP=8
PP=8
gbs=128
seq_len=4096
prof=2 # 1: profile, 2: evenpart, 3: adapipe
mem=74
world_size=64
log_dir=log
layer_num=96
hostfile=/mnt/share/hostfile

baseline_strategy=recompute_config/gpt/baseline/${PP}pp.json
python3 recompute_config/gpt/baseline/generate_baseline.py $PP $layer_num $baseline_strategy
mpirun --allow-run-as-root -n $world_size --hostfile $hostfile sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP 32 $seq_len $mem 1 $log_dir $layer_num
if [ $? -ne 0 ]; then
    mpirun --allow-run-as-root -N 1 --hostfile $hostfile pkill -f pretrain
    exit
fi

log_dir_even=${log_dir}_even
if [ ! -d $log_dir_even ]; then
    mkdir $log_dir_even
fi
python3 recompute_config/parser.py $log_dir $world_size $TP $PP $seq_len gpt
python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $seq_len $mem $layer_num 1
mpirun --allow-run-as-root -n $world_size --hostfile /mnt/nfs/app/expr/hostfile -x PATH sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP $gbs $seq_len $mem 2 $log_dir_even $layer_num | tee gpt_result/evenpart_seq${seq_len}/tp${TP}_pp${PP}_gbs${gbs}.txt
mpirun --allow-run-as-root -N 1 --hostfile /mnt/nfs/app/expr/hostfile -x PATH pkill -f pretrain

log_dir_ada=${log_dir}_adapipe
if [ ! -d $log_dir_ada ]; then
    mkdir $log_dir_ada
fi
PYTHONPATH=/mnt/share/megatron-checkpoint/recompute_config/llama2/build/lib.linux-x86_64-3.10 python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $seq_len $mem $layer_num 0
mpirun --allow-run-as-root -n $world_size --hostfile $hostfile sh examples/pretrain_gpt_distributed_mpi.sh $TP $PP $gbs $seq_len $mem 3 $log_dir $layer_num
mpirun --allow-run-as-root -N 1 --hostfile $hostfile -f pretrain
