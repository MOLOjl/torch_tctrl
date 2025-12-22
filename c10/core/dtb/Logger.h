#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <third_party/kineto/libkineto/third_party/dynolog/third_party/json/single_include/nlohmann/json.hpp>
#include <unistd.h>

#include <c10/core/dtb/comm_heads.h>

namespace c10 {
namespace dtb {

struct DTRLogger {
    std::string time_prefix;
    std::ofstream out;

    DTRLogger();

    static std::string get_time_prefix();
    std::string get_filename(const std::string& name);
    void log(const std::string& str);
    static DTRLogger& logger();
};

using json = nlohmann::json;

extern bool log_json;
extern std::string INSTRUCTION, ANNOTATION, RELEASE, PIN, TIME, ARGS, MEMORY, ALIAS, NAME, CONSTANT, VALUE, EVICT, REMAT, REMATCOST, ADDR, DEGREE, TENSOR;

void DTRLogCounts(const std::string& name, size_t counts);
void DTRLogDepAndCost(const std::string& name, size_t counts, double cost);
void DTRLogEvictEvents(const std::string& name, size_t counts);
void DTRLogMemAlloc(size_t alloc, size_t reserved);
void DTRLogEvictAPSEvents(size_t counts);
void DTRLogDestructEvents();
void DTRLogRematEvents(const std::string& name, size_t counts);
void DTRLogAddress(const std::string& name, uintptr_t addr, size_t memory);
void DTRLogTensorInfo(const std::string& name, uintptr_t addr, size_t memory, size_t degree, double cost, int device);
void DTRLogOPRecords(const int64_t& rid, const std::string& name, const int64_t& compute_cost, size_t &mem_cost, std::vector<std::string> &inputs, std::vector<std::string> &outputs, int device);
void DTRLogCalculativeRematsRecords(const int64_t& rid, const std::string& name, const int& remat_counts);
void DTRLogLifeCycle(const std::string& tag, const size_t& org, const size_t& lck, const size_t& remat);
void DTRLogConstant(const std::string& name);
void DTRLogMemory(const std::string& name, size_t memory);
void DTRLogApCost(const std::string& name, double cost);
void DTRLogAlias(const std::string& name, int index);
void DTRLogCopyFrom(const std::string& to, const std::string& from);
void DTRLogCopy(const std::string& new_name, const std::string& old_name);
void DTRLogMutate(const std::string& name, const std::vector<std::string>& args, const std::vector<size_t>& mutate, const std::string& time);
void DTRLogRelease(const std::string& name);
void DTRLogPin(const std::string& name);
void DTRLogCall(const std::vector<std::string>& res, const std::string& name, const std::vector<std::string>& args, const std::string& time);
void DTRLogMemEvents(const std::string& name, size_t size, int64_t addr);

}
}
