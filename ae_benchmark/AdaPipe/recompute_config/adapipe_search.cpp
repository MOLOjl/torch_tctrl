#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

float dp_algorithm(int left_memory, int op_num, std::vector<int> &op_list, std::vector<int> &op_mem, std::vector<float> &op_cost, py::list &final_trace) {
	float *l = (float*)malloc(sizeof(float) * (op_num + 1) * (left_memory + 1));
	int *trace = (int*)malloc(sizeof(int) * (op_num + 1) * (left_memory + 1));
	
	for (int i = 0; i < left_memory + 1; ++i) {
		l[i] = 0.0;
	}
	
	for (int op_idx = 0; op_idx < op_num; ++op_idx) {
		int num = op_list[op_idx];
		int mem = op_mem[op_idx];
		float cost = op_cost[op_idx];

		for (int m = 0; m < left_memory + 1; ++m) {
			float tmp = l[op_idx * (left_memory + 1) + m];
			int tmp_t = 0;
			for (int j = 1; j < num + 1; ++j) {
				if (m - mem * j >= 0) {
					float new_value = l[op_idx * (left_memory + 1) + m - mem * j] + cost * j;
					if (tmp < new_value) {
						tmp = new_value;
						tmp_t = j;
					}
				}
			}
			l[(op_idx + 1) * (left_memory + 1) + m] = tmp;
			trace[(op_idx + 1) * (left_memory + 1) + m] = tmp_t;
		}
	}
	
	float max_cost = l[op_num * (left_memory + 1) + left_memory];
	int f = left_memory;
	for (int i = op_num; i > 0; --i) {
		final_trace[i] = py::int_(trace[i * (left_memory + 1) + f]);
		f = f - trace[i * (left_memory + 1) + f] * op_mem[i - 1];
	}
	free(l);
	free(trace);
	return max_cost;
}


PYBIND11_MODULE(adapipe_search, m) {
	m.doc() = "dp process";
	m.def("dp_algorithm", &dp_algorithm, "A function for dp process");
}
