import sys

def parse_sample(log_dir, world_size):
    minimal_time = 1e9
    tp = 1
    while tp <= 8:
        pp = 1
        res_pp = world_size // tp
        while pp <= res_pp:
            filename = f"{log_dir}/tp{tp}_pp{pp}.txt"
            label = "elapsed time per iteration (ms):"
            iteration_time = 0
            with open(filename, 'r') as f:
                for line in f:
                    if line.find("iteration       18/      20") >= 0:
                        start_pos = line.find(label) + len(label)
                        end_pos = line.find("|", start_pos)
                        iteration_time += float(line[start_pos:end_pos])
                    if line.find("iteration       19/      20") >= 0:
                        start_pos = line.find(label) + len(label)
                        end_pos = line.find("|", start_pos)
                        iteration_time += float(line[start_pos:end_pos])
            iteration_time = iteration_time / 2
            if iteration_time > 1e-5 and iteration_time < minimal_time:
                minimal_time = iteration_time
                #print(f"filename: {filename}, minimal_time: {minimal_time}")
            pp *= 2
        tp *= 2
    return minimal_time

def parse_experiment(exper_name, world_size, config_list):
    print(f"parse {exper_name}")
    print("{:<20}".format("(gbs, seq_len)"), end="")
    print("{:<20}".format("adapipe"), end="")
    print("{:<20}".format("evenpart"))
    for gbs, seq_len in config_list:
        print("{:<20}".format(f"({gbs}, {seq_len})"), end="")
        for t in ["adapipe", "evenpart"]:
            log_dir = f"{exper_name}/gbs{gbs}_seq{seq_len}_{t}"
            iteration_time = parse_sample(log_dir, world_size)
            if iteration_time > 1e8:
                iteration_time = -1
            print("{:20}".format(f"{iteration_time:.2f}"), end="")
        print()


# parse gpt_result
exper_name = "gpt_result"
world_size = 64
config_list = [(128, 4096), (64, 8192), (32, 16384)]
parse_experiment(exper_name, world_size, config_list)

exper_name = "llama2_result"
world_size = 32
parse_experiment(exper_name, world_size, config_list)
