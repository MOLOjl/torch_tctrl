import sys

if len(sys.argv) != 3:
    print("please input directory, world_size")

log_dir = sys.argv[1]
world_size = int(sys.argv[2])

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
                if line.find("iteration       19/      20") >= 0 or line.find("iteration       18/      20") >= 0:
                    start_pos = line.find(label) + len(label)
                    end_pos = line.find("|", start_pos)
                    iteration_time += float(line[start_pos:end_pos])
        print(iteration_time / 2)
        pp *= 2
    tp *= 2
