from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
import json
import sys
import adapipe_search

class Config:
    def __init__(self,
                 micro_batch_size,
                 seq_len,
                 hidden_size,
                 ffn_hidden_size,
                 head_num,
                 num_kv_heads,
                 num_layers,
                 micro_size,
                 vocab_size,
                 mp,
                 dp,
                 pp,
                 device_memory=75,
                 profile_dict=None):
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.head_num = head_num
        self.num_kv_heads = num_kv_heads
        self.num_layers = num_layers
        self.micro_size = micro_size
        self.vocab_size = vocab_size
        self.mp = mp
        self.dp = dp
        self.pp = pp
        self.dtype_size = 2 # FP16
        self.expand_ratio = 4
        self.num_rep = self.head_num // self.num_kv_heads

        # sub attention mask
        self.use_device_memory = device_memory
        self.device_memory = device_memory * 1024 * 1024 * 1024

        # embedding compute_time
        self.profile_dict = profile_dict
        self.embedding_time = (profile_dict["embedding_forward"], profile_dict["embedding_backward"])
        self.attention_time = (profile_dict["attention_forward"], profile_dict["attention_backward"])
        self.linear_time = (profile_dict["linear_forward"], profile_dict["linear_backward"])
        self.lm_time = (profile_dict["lm_output_forward"], profile_dict["lm_output_backward"])

    def get_embedding_compute_time(self):
        return self.embedding_time

    def get_lm_output_time(self):
        return self.lm_time

    def get_attention_compute_time(self):
        # forward
        # recompute: allgather is overlaped
        return self.attention_time

    def get_linear_compute_time(self):
        return self.linear_time

    def get_embedding_memory(self):
        weight_params = self.vocab_size * self.hidden_size / self.mp
        weight_params += self.seq_len * self.hidden_size

        weight_memory = weight_params * (self.dtype_size * 2 + 4 / self.dp)
        optimizer_memory = weight_params * 4 * 2 / self.dp

        whole_weight_memory = (weight_memory + optimizer_memory)
        return whole_weight_memory

    def get_lm_output_memory(self):
        weight_params = self.vocab_size * self.hidden_size / self.mp

        weight_memory = weight_params * (self.dtype_size * 2 + 4 / self.dp)
        optimizer_memory = weight_params * 4 * 2 / self.dp
        whole_weight_memory = (weight_memory + optimizer_memory)
        return whole_weight_memory

    def get_attention_memory(self):
        weight_params = self.hidden_size * self.hidden_size // self.mp
        weight_params += self.hidden_size * self.hidden_size // self.num_rep // self.mp * 2
        weight_params += self.hidden_size * self.hidden_size // self.mp
        weight_memory = weight_params * (self.dtype_size * 2 + 4 / self.dp)
        optimizer_memory = weight_params * 4 * 2 / self.dp

        whole_weight_memory = (optimizer_memory + weight_memory)
        return whole_weight_memory

    def get_linear_memory(self):
        weight_params = self.hidden_size * self.ffn_hidden_size * 2 // self.mp
        weight_params += self.hidden_size * self.ffn_hidden_size // self.mp
        weight_memory = weight_params * (self.dtype_size * 2 + 4 / self.dp)
        optimizer_memory = weight_params * 4 * 2 / self.dp

        whole_weight_memory = (weight_memory + optimizer_memory)
        return whole_weight_memory

    def get_linear_forward_memory(self):
        forward_memory = 0.0
        # layer norm
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size
        # gather
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size / self.mp
        # mlp1
        forward_memory += self.micro_batch_size * self.seq_len * self.ffn_hidden_size * self.dtype_size * 2 / self.mp
        # swiglu
        forward_memory += self.micro_batch_size * self.seq_len * self.ffn_hidden_size * self.dtype_size / self.mp
        # dense
        forward_memory += self.micro_batch_size * self.seq_len * self.hidden_size * self.dtype_size
        # attention mask
        forward_memory += self.micro_batch_size * self.seq_len * self.seq_len * self.dtype_size
        return forward_memory

    def get_linear_operators(self):
        return {
            OperatorType.linear_norm: 1,
            OperatorType.linear_gather: 1,
            OperatorType.linear_mlp1: 1,
            OperatorType.linear_activation: 1
            }

    def get_attention_operators(self):
        return {
            OperatorType.attn_norm: 1,
            OperatorType.attn_gather: 1,
            OperatorType.attn_query: 1,
            OperatorType.attn_key: 1,
            OperatorType.attn_value: 1,
            OperatorType.attn_fa: 1
            }

class OperatorType(str, Enum):
    attn_norm = "attn_norm"
    attn_gather = "attn_gather"
    attn_query = "attn_query"
    attn_key = "attn_key"
    attn_value = "attn_value"
    attn_fa = "attn_fa"
    linear_norm = "linear_norm"
    linear_gather = "linear_gather"
    linear_mlp1 = "linear_mlp1"
    linear_activation = "linear_activation"

class IntraStageDPQuery:
    def __init__(self, config:Config, cost_map, divisor, ref_compute_time):
        self.config = config
        self.cost_map = cost_map
        self.divisor = divisor
        self.ref_compute_time = ref_compute_time

        self.cached_result = {}
        self.cached_solution = {}

    def get_intra_stage(self, stage_idx, layer_start, layer_end):
        begin_flag = layer_start == 0
        end_flag = (layer_end == self.config.num_layers * 2 + 1)

        # generate layer graph from [layer_start, layer_end]
        weight_memory = 0.0
        ftime, btime = 0.0, 0.0
        # process first embedding layer
        if layer_start == 0:
            weight_memory += self.config.get_embedding_memory()
            f, b = self.config.get_embedding_compute_time()
            ftime += f
            btime += b
            layer_start += 1

        if layer_end ==  2 * self.config.num_layers + 1:
            weight_memory += self.config.get_lm_output_memory()
            f, b = self.config.get_lm_output_time()
            ftime += f
            btime += b
            layer_end -= 1
        # process other layers
        layer_num = layer_end - layer_start + 1
        atten_num = 0
        if layer_start % 2 == 0: # begin with linear layer
            atten_num = layer_num // 2
        else:
            atten_num = (layer_num + 1) // 2

        lin_num = layer_num - atten_num

        key = f"{stage_idx}_{layer_num}"
        if layer_start % 2 == 0: # linear
            key += "_linear"
        else:
            key += "_atten"
        if begin_flag:
            key += "_begin"
        if end_flag:
            key += "_end"

        if key in self.cached_result:
            return self.cached_result[key], self.cached_solution[key]

        weight_memory += atten_num * self.config.get_attention_memory()
        f, b = self.config.get_attention_compute_time()
        ftime += atten_num * f
        btime += atten_num * b
        weight_memory += lin_num * self.config.get_linear_memory()
        f, b = self.config.get_linear_compute_time()
        ftime += lin_num * f
        btime += lin_num * b

        # left memory
        left_memory = (self.config.device_memory - weight_memory)
        left_memory -= self.config.get_linear_forward_memory()
        print(f"left memory: {left_memory / 1024 / 1024 / 1024} GB")
        gpt_config = self.config
        hidden_state = gpt_config.micro_batch_size * gpt_config.seq_len * gpt_config.hidden_size * gpt_config.dtype_size // gpt_config.mp
        # two for layernorm, one for two dropout, two for output
        left_memory -= hidden_state * 1 * layer_num * (self.config.pp - stage_idx)

        left_memory = left_memory / self.divisor / (self.config.pp - stage_idx)
        print(f"left_memory again: {left_memory}")

        # reduce default recomputation
        #left_memory -= atten_num * self.cost_map[OperatorType.LINEAR_INPUT][0]
        #if layer_start % 2 == 0: # begin with linear layer
        #    left_memory -= self.cost_map[OperatorType.LINEAR_INPUT][0]

        ## add default recompute
        #if layer_end % 2 == 1: # end with attention layer
        #    btime -= self.cost_map[OperatorType.LINEAR_INPUT][1]

        # beibao wenti
        operator_count = {
            OperatorType.attn_norm: 0,
            OperatorType.attn_gather: 0,
            OperatorType.attn_query: 0,
            OperatorType.attn_key: 0,
            OperatorType.attn_value: 0,
            OperatorType.attn_fa: 0,
            OperatorType.linear_norm: 0,
            OperatorType.linear_gather: 0,
            OperatorType.linear_mlp1: 0,
            OperatorType.linear_activation: 0}
        linear_operators = self.config.get_linear_operators()
        attention_operators = self.config.get_attention_operators()
        for op, num in linear_operators.items():
            operator_count[op] += num * (layer_num - atten_num)
        for op, num in attention_operators.items():
            operator_count[op] += num * atten_num

        # pre-test:
        test_compute_time = ftime + btime
        for op, num in operator_count.items():
            mem, cost = self.cost_map[op]
            test_compute_time -= cost * num
        if test_compute_time > self.ref_compute_time:
            return None, None

        # beibao wenti
        left_memory = int(left_memory)
        if left_memory < 0:
            return None, None
        op_num = len(operator_count)

        b = time.time()
        operator_keys = list(operator_count.keys())
        op_list = []
        op_mem = []
        op_cost = []
        for op in operator_keys:
            op_list.append(operator_count[op])
            mem, cost = self.cost_map[op]
            op_mem.append(mem)
            op_cost.append(cost)
        final_trace = [-1] * (len(operator_keys) + 1)
        max_cost = adapipe_search.dp_algorithm(left_memory, len(operator_keys), op_list, op_mem, op_cost, final_trace)
        e = time.time()
        print(f"time: {e - b} s")

        final_time = ftime + btime - max_cost
        recompute_op_dict = None
        record_time = (ftime, btime - max_cost)

        if final_time > self.ref_compute_time:
            final_time = None
            record_time = None
        else:
            # trace back
            recompute_op_dict = {}
            for i in range(op_num, 0, -1):
                recompute_op_dict[operator_keys[i - 1]] = final_trace[i]

        self.cached_result[key] = record_time
        self.cached_solution[key] = recompute_op_dict

        return record_time, recompute_op_dict

@dataclass
class UniInterStageDPItem:
    """Items used in intra stage DP algorithm.

    Attributes:
        1. main cost: the computation time and comm time
            of activations.
        2. weight cost: the comm time to all gather and
            reduce scatter gradients
        3. cur_state: (device_mesh, l_s, l_e),
            point to intra stage
        4. next_state: (prev_dn,)
            point to prev item
    """
    main_cost: float = 0.0
    next_warmup_time: float = 0.0
    next_steady_time: float = 0.0
    iter_time: float = 0.0


def stage_dp_algorithm(config:Config, intra_stage_dp_query: IntraStageDPQuery, output_filename):
    layer_num = config.num_layers * 2 + 2

    stage_max = [[None] * layer_num for _ in range(config.pp)]
    prev_pointers = [[-1] * layer_num for _ in range(config.pp)]
    stage_idx = config.pp - 1
    for i in range(layer_num - 1, -1, -1):
        result, _ = intra_stage_dp_query.get_intra_stage(stage_idx, i, layer_num - 1)
        if result is None:
            break
        iter_time = result[0] + result[1]
        new_item = UniInterStageDPItem(
            main_cost=iter_time * config.micro_size,
            iter_time=iter_time)
        new_item.next_steady_time = (config.micro_size - 2) * iter_time
        new_item.next_warmup_time = iter_time
        new_item.next_cool_time = iter_time
        if config.micro_size == 1:
            new_item.next_steady_time = iter_time
            new_item.next_warmup_time = 0
            new_item.next_cool_time = 0
        stage_max[stage_idx][i] = new_item
        #print(f"init: {new_item}")

    for s_n in range(config.pp - 2, -1, -1):
        begin = time.time()
        for l_s in range(layer_num - (config.pp - s_n), -1, -1):
            for l_m in range(l_s, layer_num - (config.pp - s_n) + 1):
                cur_item = stage_max[s_n][l_s]
                #print(f"cur_item: {cur_item}")
                next_item = stage_max[s_n + 1][l_m + 1]
                #print(f"next_item: {next_item}")
                if next_item is None:
                    continue
                mid_item, _ = intra_stage_dp_query.get_intra_stage(s_n, l_s, l_m)
                if mid_item is None:
                    break

                cur_iter_time = mid_item[0] + mid_item[1]

                next_iter_time = next_item.iter_time
                max_iter_time = max(cur_iter_time, next_iter_time)

                iter_num = config.micro_size - (config.pp - s_n)
                if iter_num >= 0:
                    steady_time = max_iter_time * iter_num
                else:
                    steady_time = next_item.next_steady_time

                # warmup stage time
                warmup_time = mid_item[0]
                if iter_num >= 0:
                    warmup_time += max(
                        mid_item[0] * (config.pp - s_n - 1),
                        next_item.next_warmup_time)
                else:
                    warmup_time += max(
                        mid_item[0] * (config.micro_size - 1),
                        next_item.next_warmup_time)

                # cool stage time
                cool_time = mid_item[1]
                if iter_num >= 0:
                    cool_time += max(
                        mid_item[1] * (config.pp - s_n - 1),
                        next_item.next_cool_time)
                else:
                    cool_time += max(
                        mid_item[1] * (config.micro_size - 1),
                        next_item.next_cool_time)

                # new main_cost
                new_main_cost = steady_time + warmup_time + cool_time
                if cur_item is None or cur_item.main_cost > new_main_cost:
                    new_item = UniInterStageDPItem(
                        main_cost=new_main_cost,
                        iter_time = max_iter_time)

                    if iter_num >= 1:
                        new_item.next_steady_time = (steady_time - max_iter_time)
                        if cur_iter_time >= max_iter_time:
                            new_item.next_warmup_time = (
                                warmup_time + mid_item[1])
                            new_item.next_cool_time = (
                                cool_time + mid_item[0])
                        else:
                            new_item.next_warmup_time = (
                                warmup_time + max_iter_time - mid_item[1])
                            new_item.next_cool_time = (
                                cool_time + max_iter_time - mid_item[0])
                    else:
                        new_item.next_steady_time = (
                            steady_time + mid_item[0] + mid_item[1])
                        new_item.next_warmup_time = warmup_time
                        new_item.next_cool_time = cool_time
                    stage_max[s_n][l_s] = new_item
                    prev_pointers[s_n][l_s] = l_m
        end = time.time()
        print(f"stage idx: {s_n}, elapsed time: {end - begin} s")

    # find the best item
    best_solution = stage_max[0][0]
    print(f"best solution: {best_solution}")

    # print result
    solution = []
    stage_idx = 0
    layer_idx = 0
    while stage_idx < config.pp:
        next_layer_idx = prev_pointers[stage_idx][layer_idx]
        if stage_idx == config.pp - 1:
            next_layer_idx = layer_num - 1
        stage_time, stage_solution = intra_stage_dp_query.get_intra_stage(stage_idx, layer_idx, next_layer_idx)
        print(f"stage idx: {stage_idx}, layer: [{layer_idx}, {next_layer_idx}] stage time: {stage_time}")
        stage_solution["stage_idx"] = stage_idx
        stage_solution["layer_range"] = [layer_idx, next_layer_idx]
        solution.append(stage_solution)
        layer_idx = next_layer_idx + 1
        stage_idx += 1

    with open(output_filename, "w") as f:
        json.dump(solution, f, indent=4)


def print_elapsed_time(config:Config,
                       intra_stage_dp_query: IntraStageDPQuery,
                       solution):
    last_solution = solution[config.pp - 1]
    layer_range = last_solution["layer_range"]
    stage_time, _ = intra_stage_dp_query.get_intra_stage(config.pp - 1, layer_range[0], layer_range[1])
    iter_time = stage_time[0] + stage_time[1]
    cur_item = UniInterStageDPItem(
        main_cost=iter_time * config.micro_size,
        iter_time=iter_time)
    cur_item.next_steady_time = (config.micro_size - 2) * iter_time
    cur_item.next_warmup_time = iter_time
    cur_item.next_cool_time = iter_time
    if config.micro_size == 1:
        cur_item.next_steady_time = iter_time
        cur_item.next_warmup_time = 0
        cur_item.next_cool_time = 0

    for stage_idx in range(config.pp - 2, -1, -1):
        stage_config = solution[stage_idx]
        layer_range = stage_config["layer_range"]
        mid_item, _ = intra_stage_dp_query.get_intra_stage(stage_idx, layer_range[0], layer_range[1])

        cur_iter_time = mid_item[0] + mid_item[1]

        next_iter_time = cur_item.iter_time
        max_iter_time = max(cur_iter_time, next_iter_time)

        iter_num = config.micro_size - (config.pp - stage_idx)
        if iter_num >= 0:
            steady_time = max_iter_time * iter_num
        else:
            steady_time = cur_item.next_steady_time

        # warmup stage time
        warmup_time = mid_item[0]
        if iter_num >= 0:
            warmup_time += max(
                mid_item[0] * (config.pp - stage_idx - 1),
                cur_item.next_warmup_time)
        else:
            warmup_time += max(
                mid_item[0] * (config.micro_size - 1),
                cur_item.next_warmup_time)

        # cool stage time
        cool_time = mid_item[1]
        if iter_num >= 0:
            cool_time += max(
                mid_item[1] * (config.pp - stage_idx - 1),
                cur_item.next_cool_time)
        else:
            cool_time += max(
                mid_item[1] * (config.micro_size - 1),
                cur_item.next_cool_time)

        # new main_cost
        new_main_cost = steady_time + warmup_time + cool_time

        new_item = UniInterStageDPItem(
            main_cost=new_main_cost,
            iter_time = max_iter_time)

        if iter_num >= 1:
            new_item.next_steady_time = (steady_time - max_iter_time)
            if cur_iter_time >= max_iter_time:
                new_item.next_warmup_time = (
                    warmup_time + mid_item[1])
                new_item.next_cool_time = (
                    cool_time + mid_item[0])
            else:
                new_item.next_warmup_time = (
                    warmup_time + max_iter_time - mid_item[1])
                new_item.next_cool_time = (
                    cool_time + max_iter_time - mid_item[0])
        else:
            new_item.next_steady_time = (
                steady_time + mid_item[0] + mid_item[1])
            new_item.next_warmup_time = warmup_time
            new_item.next_cool_time = cool_time

        cur_item = new_item
    print(cur_item)


def even_part_dp_algorithm(config:Config,
                           intra_stage_dp_query: IntraStageDPQuery,
                           stage_list,
                           output_filename):
    # print result
    solution = []
    for stage_idx in range(config.pp):
        layer_range = stage_list[stage_idx]
        stage_time, stage_solution = intra_stage_dp_query.get_intra_stage(stage_idx, layer_range[0], layer_range[1])
        stage_solution["stage_idx"] = stage_idx
        stage_solution["layer_range"] = layer_range
        solution.append(stage_solution)
        stage_idx -= 1
    print_elapsed_time(gpt_config, intra_stage_dp_query, solution)

    with open(output_filename, "w") as f:
        json.dump(solution, f, indent=4)

def gcd(a, b):
    while b != 0:
        temp = a % b
        a = b
        b = temp
    return a


if len(sys.argv) != 8:
    print("please input world_size, tp, pp, global batch, seq_len, mem, whether even part ")

world_size = int(sys.argv[1])
tp = int(sys.argv[2])
pp = int(sys.argv[3])
gbs = int(sys.argv[4])
seq_len = int(sys.argv[5])
mem = int(sys.argv[6])
layer_num = int(sys.argv[7])
even = int(sys.argv[8])
dp = world_size // tp // pp

hidden_size=8192

with open(f"recompute_config/llama2/profile_result_{tp}tp_{pp}pp_{seq_len}seq_{hidden_size}hidden.json", 'r') as f:
    profile_dict = json.load(f)

gpt_config = Config(
    micro_batch_size=1,
    seq_len=seq_len,
    hidden_size=8192,
    ffn_hidden_size=28672,
    head_num=64,
    num_kv_heads=8,
    num_layers=layer_num,
    micro_size=gbs//dp,
    vocab_size=50432,
    mp=tp,
    dp=dp,
    pp=pp,
    device_memory=mem,
    profile_dict=profile_dict)

hidden = gpt_config.micro_batch_size * gpt_config.seq_len * gpt_config.hidden_size * gpt_config.dtype_size // gpt_config.mp
lse_size = gpt_config.head_num // gpt_config.mp * gpt_config.seq_len * gpt_config.dtype_size
mapping0 = gpt_config.micro_batch_size * gpt_config.seq_len * gpt_config.ffn_hidden_size * gpt_config.dtype_size // gpt_config.mp
hidden_kv = hidden * gpt_config.num_kv_heads // gpt_config.head_num
# cost of GPT3 model, mp = 4
cost_map = {
    OperatorType.attn_norm: (hidden, profile_dict[OperatorType.attn_norm]),
    OperatorType.attn_gather: (hidden * gpt_config.mp, profile_dict[OperatorType.attn_gather]),
    OperatorType.attn_query: (hidden, profile_dict[OperatorType.attn_query]),
    OperatorType.attn_key: (hidden_kv, profile_dict[OperatorType.attn_key]),
    OperatorType.attn_value: (hidden_kv, profile_dict[OperatorType.attn_value]),
    # out_padded is the same as the output
    OperatorType.attn_fa: (hidden + lse_size, profile_dict[OperatorType.attn_fa]),
    OperatorType.linear_norm: (hidden, profile_dict[OperatorType.linear_norm]),
    OperatorType.linear_gather: (hidden * gpt_config.mp, profile_dict[OperatorType.linear_gather]),
    OperatorType.linear_mlp1: (mapping0 * 2, profile_dict[OperatorType.linear_mlp1]),
    OperatorType.linear_activation: (mapping0, profile_dict[OperatorType.linear_activation]),
}

max_divisor = gcd(mapping0, hidden)
max_divisor = gcd(max_divisor, hidden_kv)
max_divisor = gcd(max_divisor, mapping0)
for op, (gmem, cost) in cost_map.items():
    cost_map[op] = (gmem // max_divisor, cost)
print(f"{max_divisor=}")

# get reference running time
ref_layer_num = gpt_config.num_layers // gpt_config.pp

ref_compute_time = (sum(gpt_config.get_attention_compute_time()) + sum(gpt_config.get_linear_compute_time())) * ref_layer_num
first_stage = ref_compute_time + sum(gpt_config.get_embedding_compute_time())
last_stage = ref_compute_time + sum(gpt_config.get_lm_output_time())
ref_compute_time = max(first_stage, last_stage)
print(f"ref iteration time: {ref_compute_time}")

intra_stage_dp_query = IntraStageDPQuery(gpt_config, cost_map, max_divisor, ref_compute_time)

if even <= 0:
    output_filename = f"recompute_config/llama2/adapipe/gpt_{gpt_config.seq_len}seq_{gpt_config.mp}mp_{gpt_config.pp}pp_{gbs}gbs_{mem}mem.json"
    stage_dp_algorithm(gpt_config, intra_stage_dp_query, output_filename)
else:
    layer_per_stage = gpt_config.num_layers // gpt_config.pp
    output_filename = f"recompute_config/llama2/evenpart/gpt_{gpt_config.seq_len}seq_{gpt_config.mp}mp_{gpt_config.pp}pp_{mem}mem.json"
    stage_list = []
    for i in range(gpt_config.pp):
        start = i * layer_per_stage * 2
        if i > 0:
            start += 1
        end = (i + 1) * layer_per_stage * 2
        if i == gpt_config.pp - 1:
            end += 1
        stage_list.append([start, end])
    even_part_dp_algorithm(gpt_config, intra_stage_dp_query, stage_list, output_filename)
