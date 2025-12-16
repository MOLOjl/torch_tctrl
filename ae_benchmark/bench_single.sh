#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# cd Llama_sft
# bash bench_llama.sh ./deepspeed_single_node.yaml 0,1,2,3,4,5,6,7
# cd ..
# cd ViT
# bash bench_vit.sh ours
# cd..
cd alphafold
bash bench_openfold.sh
cd..