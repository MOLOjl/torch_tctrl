import sys
import json

if len(sys.argv) != 4:
    print(f"please input pp, layer_num, output_filename")
    exit(0)

pp = int(sys.argv[1])
layer_num = int(sys.argv[2])
output_filename = sys.argv[3]
if layer_num % pp != 0:
    print("layer_num cannot divide pp")
    exit(0)

layer_per_pp = layer_num // pp

solution = []

layer_start = 0
for i in range(pp):
    s = {}
    ln = layer_per_pp * 2
    if i == 0:
        ln += 1
    if i == pp - 1:
        ln += 1

    layer_end = layer_start + ln - 1
    s["layer_range"] = [layer_start, layer_end]
    layer_start = layer_end + 1
    solution.append(s)

with open(output_filename, "w") as f:
    json.dump(solution, f, indent=4)
