#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# profile
# world_size=4
# TP=1
world_size=8
TP=2
PP=4
gbs=64
seq_len=8192
hidden_size=6144
head_num=48

prof=2 # 1: profile, 2: evenpart, 3: adapipe
mem=35
layer_num=24

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

baseline_strategy=${baseline_strategy_dir}/${layer_num}ln_${PP}pp.json
python3 recompute_config/gpt/generate_baseline.py $PP $layer_num $baseline_strategy

result_dir=gpt_result
if [ ! -d $result_dir ]; then
    mkdir $result_dir
fi
profile_dir=${result_dir}/gbs${gbs}_seq${seq_len}_profile
if [ ! -d $profile_dir ]; then
    mkdir $profile_dir
fi
log_dir=${profile_dir}/log_tp${tp}_pp${pp}
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi
result_name=${profile_dir}/tp${tp}_pp${pp}.txt

mpirun -n $world_size sh examples/pretrain_single_test.sh $TP $PP 32 $seq_len $layer_num $hidden_size $head_num $mem 1 $log_dir | tee $result_name
if [ $? -ne 0 ]; then
    exit
fi

python3 recompute_config/parser.py gpt $log_dir $world_size $TP $PP $seq_len $hidden_size

evenpart_log=gpt_result/gbs${gbs}_seq${seq_len}_evenpart
if [ ! -d $evenpart_log ]; then
    mkdir $evenpart_log
fi
log_dir=${evenpart_log}/log_tp${tp}_pp${pp}
result_name=${evenpart_log}/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

PYTHONPATH=recompute_config python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $mem $seq_len $layer_num $hidden_size $head_num 1
mpirun -n $world_size sh examples/pretrain_single_test.sh $TP $PP $gbs $seq_len $layer_num $hidden_size $head_num $mem 2 $log_dir | tee $result_name

adapipe_log=gpt_result/gbs${gbs}_seq${seq_len}_adapipe
if [ ! -d $adapipe_log ]; then
    mkdir $adapipe_log
fi
log_dir=${adapipe_log}/log_tp${tp}_pp${pp}
result_name=${adapipe_log}/tp${tp}_pp${pp}.txt
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi
PYTHONPATH=recompute_config python3 recompute_config/gpt/search.py $world_size $TP $PP $gbs $mem $seq_len $layer_num $hidden_size $head_num 0
mpirun -n $world_size sh examples/pretrain_single_test.sh $TP $PP $gbs $seq_len $layer_num $hidden_size $head_num $mem 3 $log_dir | tee $result_name
