tokenizer_path=/data/wangzehua/model_space/Llama-2-13b-hf/tokenizer.model
dataset_path=/data/wangzehua/dataset/oscar-en-10k/oscar-en-10k

python preprocess_data.py \
--input $dataset_path/oscar-en-10k.jsonl \
--output-prefix $dataset_path/oscar-en-10k-meg-llama \
--tokenizer-type Llama2Tokenizer \
--tokenizer-model $tokenizer_path \
--workers 16 \
--append-eod