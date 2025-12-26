#!/bin/bash
#SBATCH -J llama_large            # 作业名
#SBATCH -p gpu                    # 队列/分区（按你集群实际情况改）
#SBATCH -N 64                      # 节点数（示例：2 节点）
#SBATCH --gres=gpu:4              # 每节点 GPU 数
#SBATCH --ntasks-per-node=1       # 每节点只启动 1 个 task（你自己用 torchrun）
#SBATCH --cpus-per-task=16        # CPU 核数（按集群建议）
#SBATCH --time=04:00:00           # 运行时间
#SBATCH -o %x-%j.out               # stdout
#SBATCH -e %x-%j.err               # stderr

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

export SLURM_CPU_BIND=none
export SLURM_GPU_BIND=none

module purge
module load compilers/gcc/12.2.0 \
            compilers/cuda/11.8 \
            cudnn/8.6.0.163_cuda11.x \
            nccl/2.11.4-1_cuda11.8

source activate dtb

echo "JobID: $SLURM_JOB_ID"
echo "Node list:"
scontrol show hostnames $SLURM_NODELIST
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

bash ../mn_pretrain_llama3_distributed.sh large
