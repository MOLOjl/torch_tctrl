# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

### 单卡fp16， 多卡bf16

model_path = "/data/wangzehua/model_space/Llama-2-7b-hf"
# model_path = "/data/wangzehua/model_space/chatglm3-6b-base"       # chatglm with different weights names, need more tinker
# model_path = "/share/datasets/public_models/Llama-2-7b-hf"
dataset_path= "/data/wangzehua/dataset/stack-exchange-paired"
# dataset_path= "/share/wangzehua/dtr_workspace/hf_train/stack-exchange-paired"
RECORD_MEM_SNAPSHOT = True if os.environ.get('RECORD_MEM_SNAPSHOT') == '1' else False
TORCH_TRACER = True if os.environ.get('TORCH_TRACER') == '1' else False
MULTI_GPU = True if os.environ.get('MULTI_GPU') == '1' else False
USE_DTR =  True if os.environ.get('DTR_ENABLE') == '1' else False
LORA_ENABLE =  True if os.environ.get('LORA_ENABLE') == '1' else False
snapshot_filename = os.environ.get('SNAP_FILE_NAME')
MAX_STEPS = 100
WARM_STEPS = 10
if RECORD_MEM_SNAPSHOT:
    MAX_STEPS = 6
    WARM_STEPS = 1



@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=model_path, metadata={"help": "the model name"})

    dataset_name: Optional[str] = field(default=dataset_path, metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output_model",
            max_steps=MAX_STEPS,
            logging_steps=10,
            save_steps=120 if MULTI_GPU else 10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False,
            group_by_length=False,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=WARM_STEPS,
            weight_decay=0.05,
            optim="paged_adamw_32bit",
            bf16=True if MULTI_GPU else False,                                          # bf16 without unscale error
            remove_unused_columns=False,
            run_name="sft_llama2",
            report_to="wandb",
            use_dtr=USE_DTR
        )
    )

    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    )


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    string += ' | fragmentation: {}'.format(
        (torch.cuda.max_memory_reserved() - torch.cuda.max_memory_allocated()) / torch.cuda.max_memory_reserved()
    )
    print("[Rank {}] {}".format(torch.distributed.get_rank() if MULTI_GPU else 0, string),                                     # TODO: 依据是否并行采用 torch.distributed.get_rank()
            flush=True)

torch.random.manual_seed(42)

script_args = tyro.cli(ScriptArguments)
mem_budget = script_args.training_args.mem_budget
if USE_DTR:
    torch.init_dtb_manager()
    if mem_budget > 0:
        torch.set_memory_budget(int(mem_budget * 1e10))
    # if torch.cuda.is_available():
    #     torch.cuda.memory._set_allocator_settings('expandable_segments:True')

if script_args.training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if script_args.training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        # train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    # 这里的dataset不是tensor，不是torch的东西
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        shuffle=False,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset

if RECORD_MEM_SNAPSHOT:
    torch.cuda.memory._record_memory_history()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if MULTI_GPU:
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        # quantization_config=bnb_config,
        # device_map={"": Accelerator().local_process_index},       # TODO: What is device_map?
        trust_remote_code=True,
        use_auth_token=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        # quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},       # single gpu without this can incur numbers of H2D memcpy
        trust_remote_code=True,
        use_auth_token=True,
    )
base_model = base_model.to(torch.bfloat16) if MULTI_GPU else base_model.half()   # .half() .to(torch.bfloat16) default float, double memory
base_model.config.use_cache = False
if USE_DTR:
    print("trying to warp all parameters.")     # 这里进行前必须得先初始化了dtb系统
    base_model._apply(lambda v: v.detach().checkpoint(True))  # for fix move version
    # base_model._apply(lambda v: v.detach().checkpoint())
    print("successfully warp all parameters.")


peft_config = script_args.peft_config

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

training_args = script_args.training_args

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

if USE_DTR:
    # lock_test_path = '/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_sp_nodes.txt'
    lock_test_path = os.environ.get('LOCK_TEST_PATH', '')
    if len(lock_test_path) > 0:
        torch.load_fix_tids(lock_test_path)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config if LORA_ENABLE else None,              # enable lora train
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError, record
error_handler = get_error_handler()
error_handler.initialize()
try:
    if TORCH_TRACER:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            trainer.train()

        prof.export_chrome_trace("llama_trace.json")
    else:
        trainer.train()
    report_memory("[training summary]")
    # trainer.save_model(script_args.training_args.output_dir)
except ChildFailedError as e:
   _, failure = e.get_first_failure()
   error_handler.dump_error_file(failure.error_file, failure.exitcode)
   raise
except Exception as e:
   print('[Exception]', str(e))
#    error_handler.record(e)
   raise


# output_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
# trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
# model = AutoPeftModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(script_args.training_args.output_dir, "final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)

# except Exception as e:
#     print(f"Caught an exception: {e}")
if RECORD_MEM_SNAPSHOT:
    torch.cuda.memory._dump_snapshot(snapshot_filename)
    # res = torch.cuda.memory._snapshot()
    # with open('./sft_snapshot.txt', 'w') as f:
    #     f.write(str(res))

if USE_DTR:
    torch.log_dtr_statics()

# accelerate config
# accelerate launch ./sft_llama2.py --training_args.output_dir="/data/wangzehua/output_model"

r"""
accelerate launch sft_llama2.py \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False
"""