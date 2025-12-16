/*
  文件开头的宏定义是用于不同优化或者debug模式进行开关的
*/

#include <ATen/CheckpointTensorImpl.h>
// #include <c10/core/CheckpointTensorImpl.h>
// #include <ATen/Logger.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/dtb/DTBManager.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <string>
#include <random>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unistd.h>


namespace at {


// using c10::dtb::STATS;
using c10::dtb::getDTBPoolManager;
using c10::dtb::use_log_;
using c10::dtb::base_compute_time_;
using c10::dtb::remat_compute_time_;
using c10::dtb::search_time_;
using c10::dtb::cost_time_;
using c10::dtb::use_profile_;

using c10::dtb::base_compute_time_;
using c10::dtb::remat_compute_time_;
using c10::dtb::search_time_;
using c10::dtb::cost_time_;
using c10::dtb::use_log_;
using c10::dtb::use_profile_;
#ifdef DEBUG_MODE
using c10::dtb::record_er_counts;        // 驱逐&重物化次数
using c10::dtb::record_mem_addr;         // 是否记录内存地址
using c10::dtb::record_op_recs;          // 是否记录op历史
using c10::dtb::record_cpevict_recs;
using c10::dtb::record_fragmentation;    // 记录碎片化和内存占用数据
using c10::dtb::record_lifecycle;        // 记录ap生命周期计数分布
using c10::dtb::record_ap_cost;          // 记录ap的cost分布
using c10::dtb::record_dependcy;
using c10::dtb::record_key_chain;
using c10::dtb::current_if_any_evicted;

using c10::dtb::evict_counts;
using c10::dtb::tensor_evict_counts;
using c10::dtb::remat_counts;
using c10::dtb::cannot_evict_counts;
using c10::dtb::destruct_counts;
using c10::dtb::tensor_destruct_counts;

#endif
/**
 * Interface expose to at native
 * Kinds of log methods
 * Initiative methods in Aten
*/
#pragma region InterfaceForAtenNative

namespace native {

/**
 * This function is used to transfer the original tensor created by torch
 * and attain the ability to manage their lifecyle.
 * 
 * @param t: original tensor
 * @param if_weight: if this tensor is a weight tensor or you do not wish it to be destroyed
 * 
 * @return res: tensor returned is a "empty" tensor without storage, which is moved into cpti
 * 
 * 
 * @skip
 * ---------------------------------------some debug notes-------------------------------------
 * !TODO: 潜在问题 引入张量t有undefined的情况，此种情况cpti构造出来的tensor却不是undefined的了，可以额外标记tensor是undefined
 * 需要规避空tensor带来的可能问题，调用empty tensor的成员方法会报错
 * [TAG] if_weight的标记操作只是保证了生命周期的合理性，即权重不释放，非权重的无源tensor释放
*/
Tensor checkpoint(Tensor& t, bool if_weight) {
  // STATS.track("checkpoint");
  auto cpti = c10::make_intrusive<CheckpointTensorImpl>(t, if_weight);      // cpti->ref->value->value->t 是包裹的unique_ptr<Tensor> unsafeGetTensorCell()

#ifdef DEBUG_MODE
  if(c10::dtb::record_p2ap_actions) 
    std::cout << "[checkpoint INSERT AP PTR]" << reinterpret_cast<void*>(cpti->unsafeGetTensorCell()->pool->addr) << "\n";
  if (use_log_) {
    c10::dtb::DTRLogConstant(cpti->counter_name());
    c10::dtb::DTRLogMemory(std::to_string(cpti->unsafeGetTensorCell()->pool->if_weight)+"-"+std::to_string(if_weight), cpti->unsafeGetTensorCell()->memory());
  }
#endif
  auto res = Tensor(cpti);
  return res;
}

Tensor fake_checkpoint(const Tensor& t) {
  // STATS.track("checkpoint");
  // if(!t.defined())
  //   return Tensor(nullptr);
  // auto cpti = intrusive_ptr<CheckpointTensorImpl>::make(t);   // 调用了Ref<intrusive_ptr<External>> External CheckpointTensorCell的相关构造函数
  Tensor t_ = t;
  auto cpti = c10::make_intrusive<CheckpointTensorImpl>(t_);      // cpti->ref->value->value->t 是包裹的unique_ptr<Tensor> unsafeGetTensorCell()
#ifdef DEBUG_MODE
  if (use_log_) {
    c10::dtb::DTRLogConstant(cpti->counter_name());
    c10::dtb::DTRLogMemory(std::to_string(cpti->unsafeGetTensorCell()->pool->if_weight)+"-"+std::to_string(0), cpti->unsafeGetTensorCell()->memory());
  }
#endif
  auto res = Tensor(cpti);
  return res;
}

Tensor uncheckpoint(const Tensor& t) {
  // STATS.track("uncheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  return cpti->unsafeGetTensorCell()->get();
}

/**
 * 支持上层操作的下层实现接口
*/
void pin(Tensor& t) {
  // STATS.track("pin");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(cpti != nullptr);
  cpti->unsafeGetTensorCell()->pin();
}

/**
 * Decheckpoint will create a shallow copy(detach) of tensor in cptc, which shares the same memory with cptc.
 * It can be used in those scenario where progress is not managed by DTR runtime like custom kernel and communication progress.
 * 
 * @param t: tensor self
 * @param if_comm: mark if this tensor is decheckpointed for communication
 * 
 * @return res: if t is a cptc, return inner t.detach(), otherwise return t self 
 * 
 * @skip
 * ---------------------------------------some debug notes-------------------------------------
 * 
*/
Tensor decheckpoint(const Tensor& t, bool if_comm) {
  // STATS.track("decheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpti){
    if (if_comm) cpti->unsafeGetTensorCell()->pool->is_retain = true;
    auto res = cpti->unsafeGetTensorCell()->get();
    // return res;    // BUG: segmentation fault
    return res.detach();
  }else
    return t;
  // return cpti ? cpti->unsafeGetTensorCell()->get() : t;
}

void cpti_decrease(const Tensor& t) {
  // STATS.track("decheckpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  if(cpti) cpti->release_resources();
}

bool is_checkpoint(const Tensor& t) {
  // STATS.track("is_checkpoint");
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti != nullptr;
}

void display_dtb_data(const Tensor& t) {
  if(is_checkpoint(t)) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    auto cptc = cpti->ref->value->value;
  }else{
    printf("This is not a checkpoint tensor.\n");
  }
}

Tensor try_checkpoint(Tensor& t) {
  // STATS.track("try_checkpiont");
  return is_checkpoint(t) ? t : checkpoint(t);
}

void new_log(std::string str) {
  c10::dtb::DTRLogger::logger().out = std::ofstream(c10::dtb::DTRLogger::logger().get_filename(str));
}

void annotate_log(c10::string_view str_) {
  auto str = std::string(str_);
  if (!use_log_) { return; }
  if (c10::dtb::log_json) {
    c10::dtb::json j;
    j["INSTRUCTION"] = "ANNOTATE";
    j["ANNOTATION"] = str;
    c10::dtb::DTRLogger::logger().log(j.dump());
  } else {
    c10::dtb::DTRLogger::logger().log("# " + str);
  }
}

void toggle_log(bool b) {
  use_log_ = b;
}

void clear_checkpointpool(long device, bool last_iter) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->clear_checkpointpool(device, last_iter);
#else
  while (likely(!pool.exts.empty())) {
    if (auto e = pool.exts.back().lock()) {
      e->value->pin();
    }
    pool.exts.pop_back();
  }
#endif
}

void proactive_recovery(long device, double depth, bool erase) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->proactive_remat(device, depth, erase);
#endif
}

void check_current_exts(long device){
  auto *pm = getDTBPoolManager();
  pm->pool_cur_mem_snapshot(device);
}

void init_dtb_manager(){
#ifdef MULTI_MODE
  c10::dtb::lazyInitDTB();
#endif
}

void unset_memory_budget() {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->unset_memory_budget();
#else
  pool.has_memory_budget = false;
#endif
}

void set_memory_budget(long budget) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->set_memory_budget(budget);
  c10::dtb::set_global_memory_budget(budget);
#else
  pool.memory_budget = budget;
  pool.has_memory_budget = true;
#endif
}

void register_stream(c10::Stream stream, long label) {
  c10::dtb::registerStreamLabel(stream, label);
}

void set_reserved(){
  reserved_range = true;
}

void unset_reserved(){
  reserved_range = false;
}

void set_backward_flag(){
// #ifdef MULTI_MODE
//   auto *pm = getDTBPoolManager();
//   pm->set_during_backward(true);
// #else
  during_backward = true;
#ifdef DEBUG_MODE
  if(record_op_recs) {
    c10::dtb::DTRLogAlias("begin_backward", 1);
  }
#endif
// #ifdef PROACTIVE_REMAT //[deprecated]
  // auto *pm = getDTBPoolManager();
  // pm->push_batch_evicted_tensors(c10::cuda::current_device());
// #endif
// #endif
  // printf("SET_BACKWARD_FALG TRIGGER\n");
}

void mark_layer_action(long mode, c10::string_view tag) {
  switch (mode)
  {
  case 0:
    c10::dtb::DTRLogTag("begin_layer", std::string(tag), 0);
    break;
  case 1:
    c10::dtb::DTRLogTag("end_layer", std::string(tag), 1);
    break;
  default:
    break;
  }
  /// add more action
}

void mark_tid_before_inference(){

  
}


void load_fix_tids(c10::string_view str_) {
  auto str = std::string(str_);
  auto *pm = getDTBPoolManager();
  pm->load_fix_tids(str);
}

void unset_backward_flag(){
// #ifdef MULTI_MODE
//   auto *pm = getDTBPoolManager();
//   pm->set_during_backward(false);
// #else
  during_backward = false;
#ifdef DEBUG_MODE
  if(record_op_recs) {
    c10::dtb::DTRLogAlias("end_backward", 0);
  }
#endif
#ifdef DCR_MANAGE
  c10::dtb::CheckpointTensorCell::reset_pool_counter();
#endif
#ifdef DEBUG_MODE
  if(RENUMBER_COUNTER) {
    c10::dtb::tid_counter = c10::dtb::tid_before_infer;
  }
#endif
}

void clear_batched_records(long device) {
#ifdef PROACTIVE_REMAT
  auto *pm = getDTBPoolManager();
  pm->clear_recorded_batch(device);
#endif
}

void mark_train(bool flag){
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->set_train_mode(flag);
#else
  if_train_mode = flag;
#endif
}

/// TODO: use dtb, useless here
void force_evict(long mode){
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->force_evict(0, mode);
#else
  pool.force_evict(mode);
#endif
}

void log_dtr_statics(){
#ifdef DEBUG_MODE
#ifdef MULTI_MODE
  if(record_fragmentation){
    auto *pm = getDTBPoolManager();
    int did = 0;
    for(const auto& mem_info: pm->get_peak_memory()){
      std::stringstream log_str;
      c10::dtb::DTRLogCounts("device-"+std::to_string(did)+" peak allocated memory", mem_info.first);
      c10::dtb::DTRLogCounts("device-"+std::to_string(did)+" peak reserved memory", mem_info.second);
      c10::dtb::DTRLogApCost("device-"+std::to_string(did)+" fragmentation ratio", (1. - (static_cast<double>(mem_info.first) / static_cast<double>(mem_info.second))) / 1e7);
      did++;
    }
  }
#endif
  if(record_er_counts){
    // DTRLogCounts("memory budget", memo);
    c10::dtb::DTRLogCounts("evict counts", evict_counts);
    c10::dtb::DTRLogCounts("evict tensor counts", tensor_evict_counts);
    c10::dtb::DTRLogCounts("cannot evict counts", cannot_evict_counts);
    c10::dtb::DTRLogCounts("destruct counts", destruct_counts);
    c10::dtb::DTRLogCounts("destruct tensor counts", tensor_destruct_counts);
    c10::dtb::DTRLogCounts("remat counts", remat_counts);
  }
#endif
}

/// TODO: use dtb
void toggle_sampling(bool sample) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->toggle_sampling(sample);
#else
  pool.sample_tensors = sample;
#endif
}

/// TODO: as pool member function
void toggle_ignore_small_tensors(bool ignore) {
#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
  pm->toggle_ignore_small_tensors(ignore);
#else
  pool.ignore_small_tensors = ignore;
#endif
}

void reset_profile() {
  base_compute_time_ = 0;
  remat_compute_time_ = 0;
  search_time_ = 0;
  cost_time_ = 0;
}

void toggle_profile(bool profile) {
  use_profile_ = profile;
}

long compute_time() {
  return base_compute_time() + remat_compute_time();
}

long cost_time() {
  return cost_time_;
}

long search_time() {
  return search_time_;
}

long remat_compute_time() {
  return remat_compute_time_;
}

long base_compute_time() {
  return base_compute_time_;
}

long loop_time() {
  return search_time() - cost_time();
}

}
#pragma endregion
}

