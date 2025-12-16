python  ./tools/checkpoint/convert.py \
--model-type GPT \
--loader llama2_hf \
--load-dir /data/wangzehua/model_space/Llama-2-7b-hf \
--save-dir /data/wangzehua/model_space/Llama-2-7b-ml \
--tokenizer-model /data/wangzehua/model_space/Llama-2-7b-hf \
--megatron-path /data/wangzehua/Megatron-LM/Megatron-LM \
--target-tensor-parallel-size 2 \
--target-pipeline-parallel-size 2