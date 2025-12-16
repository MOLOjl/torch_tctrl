# profile
TP=4
PP=8
gbs=64
seq_len=4096
prof=2 # 1: profile, 2: evenpart, 3: adapipe
mem=70
layer_num=80
world_size=32
log_dir=log

baseline_strategy=recompute_config/llama2/baseline/${PP}pp.json
#python3 recompute_config/llama2/baseline/generate_baseline.py $PP $layer_num $baseline_strategy

#mpirun --allow-run-as-root -n $world_size --hostfile /mnt/share/hostfile3 sh examples/pretrain_llama2_distributed_mpi.sh $TP $PP 64 $seq_len $mem 1 $log_dir $layer_num
#python3 recompute_config/parser.py log $world_size $TP $PP $seq_len llama2
#python3 recompute_config/llama2/search.py $world_size $TP $PP $gbs $seq_len $mem 1
#mpirun --allow-run-as-root -n $world_size --hostfile /mnt/share/hostfile sh examples/pretrain_llama2_mpi.sh $TP $PP $gbs $seq_len $mem 2 | tee llama2_result/evenpart_seq${seq_len}/tp${TP}_pp${PP}_gbs${gbs}.txt

PYTHONPATH=/mnt/share/megatron-checkpoint/recompute_config/llama2/build/lib.linux-x86_64-3.10 python3 recompute_config/llama2/search.py $world_size $TP $PP $gbs $seq_len $mem $layer_num 0
#mpirun -n $world_size --hostfile /mnt/nfs/app/expr/hostfile -x PATH sh examples/pretrain_llama2_mpi.sh $TP $PP $gbs $seq_len $mem 3 | tee llama2_result/adapipe_seq${seq_len}/tp${TP}_pp${PP}_gbs${gbs}.txt
