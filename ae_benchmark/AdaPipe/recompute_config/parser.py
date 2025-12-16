import sys
import os

from numpy import mean
import json

def get_profile_dict(filename):
    profile_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            # skip useless lines
            if line.find("rank") >= 0:
                continue

            # parse the line
            label = "root:"
            start_pos = line.find(label) + len(label)
            end_pos = line.find(":", start_pos)
            word = line[start_pos:end_pos]

            value = float(line[end_pos+1:].strip())
            if word in profile_dict:
                profile_dict[word].append(value)
            else:
                profile_dict[word] = [value]
    return profile_dict

def parse_forward_backward(profile_dict, label_list, final_dict):
    for label in label_list:
        forward_start_label = f"before {label} forward"
        forward_end_label = f"after {label} forward"
        forward_start_list = profile_dict[forward_start_label]
        forward_end_list = profile_dict[forward_end_label]
        l = min(len(forward_start_list), len(forward_end_list))
        time_list = [forward_end_list[i] - forward_start_list[i] for i in range(l)]
        final_dict[f"{label}_forward"] = mean(time_list[40:80]) * 1000

        backward_start_label = f"after {label} backward"
        backward_end_label = f"before {label} backward"
        backward_start_list = profile_dict[backward_start_label]
        backward_end_list = profile_dict[backward_end_label]
        l = min(len(backward_start_list), len(backward_end_list))
        time_list = [backward_end_list[i] - backward_start_list[i] for i in range(l)]
        final_dict[f"{label}_backward"] = mean(time_list[40:80]) * 1000


if len(sys.argv) != 8:
    print("Usage: python parser.py log_dir world_size tp pp seq_len model_name")

model_name = sys.argv[1]
dirname = sys.argv[2]
world_size = int(sys.argv[3])
tp = int(sys.argv[4])
pp = int(sys.argv[5])
seq_len = int(sys.argv[6])
hidden_size = int(sys.argv[7])

min_rank = 0
max_rank = world_size - 1

# get info from min_rank
filename = os.path.join(dirname, f"rank_{min_rank}.log")
print(filename)
profile_dict = get_profile_dict(filename)
print(profile_dict.keys())

final_dict = {}
label_list = ["embedding", "attention", "linear"]
parse_forward_backward(profile_dict, label_list, final_dict)

# single operator profile
op_list = [
    "attn_norm",
    "attn_gather",
    "attn_query",
    "attn_key",
    "attn_value",
    "attn_fa",
    "linear_norm",
    "linear_gather",
    "linear_mlp1",
    "linear_activation"
]

for op_name in op_list:
    final_dict[op_name] = mean(profile_dict[op_name][200:400]) * 1000

# get info for last layer
filename = os.path.join(dirname, f"rank_{max_rank}.log")
profile_dict = get_profile_dict(filename)
print(profile_dict.keys())

# lm_output

label = "lm_output"
backward_start_label = f"after {label} backward"
backward_end_label = f"before {label} backward"
backward_start_list = profile_dict[backward_start_label]
backward_end_list = profile_dict[backward_end_label]
l = min(len(backward_start_list), len(backward_end_list))
time_list = [backward_end_list[i] - backward_start_list[i] for i in range(l)]
backward_time = mean(time_list[20:50]) * 1000

forward_start_label = f"before {label} forward"
forward_start_list = profile_dict[forward_start_label]
l = min(len(backward_end_list), len(forward_start_list))
time_list = [backward_end_list[i] - forward_start_list[i] for i in range(l)]
whole_time = mean(time_list[20:50]) * 1000
final_dict[f"{label}_forward"] = whole_time - backward_time
final_dict[f"{label}_backward"] = backward_time

label_list = ["lm_output"]
parse_forward_backward(profile_dict, label_list, final_dict)

output_filename = f"recompute_config/{model_name}/profile_result_{tp}tp_{pp}pp_{seq_len}seq_{hidden_size}hidden.json"
with open(output_filename, 'w') as f:
    json.dump(final_dict, f, indent=4)
