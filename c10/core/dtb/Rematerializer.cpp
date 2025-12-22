#include <c10/core/dtb/Rematerializer.h>
#include <c10/core/dtb/utils.h>
#include <c10/core/dtb/CheckpointTensorCell.h>
#include <c10/cuda/dtb/DTBManager.h>

#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

namespace c10 {
namespace dtb {

/**
 * Methods of Rematerializer
 * Implementation about remat methods and several methods of packing and unpacking
*/
#pragma region RematerializerMethods

Rematerializer::Rematerializer(const Unsafe&,
                const rematerialize_function_t& func,
                const strongs& inputs,
                duration_t compute_cost)  :
  func(func),
  inputs(inputs),
  compute_cost(compute_cost) {
}

Rematerializer::Rematerializer(const Unsafe&,
                const rematerialize_function_t& func,
                const strongs& inputs,
                int64_t rid,
                duration_t compute_cost)  :
  func(func),
  inputs(inputs),
  rid(rid),
  compute_cost(compute_cost) {
}

void Rematerializer::release_resources() {
  func = rematerialize_function_t();
  inputs.clear();
  outputs.clear();
  ecn.reset();
}


void Rematerializer::remat() {
#ifdef DEBUG_MODE
  if(record_er_counts){
    remat_counts += 1;
  }
  // if(remat_counts>1e5)
  //   throw std::runtime_error("Remat progress has been trapped in dead loop");
#endif
  // NOTE: author thinks that refactor using RAII for exception safety. however, RAII is not suitable for remat
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  Tensors ts = uncheckpoint(inputs);
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif

#ifdef ORIG_EVICT
  if(COST_FIRST_EVICT){
    #ifdef MULTI_MODE
      auto device_id = static_cast<int>(ts[0].device().index());
      auto *pm = getDTBPoolManager();
    #endif

    #ifdef MINIMAL_EVICT_COST
      #ifdef MULTI_MODE
      pm->auto_evict(device_id, memory_cost_records[rid]);
      #else
      pool.auto_evict(memory_cost_records[rid]);
      #endif
    #endif
  }
#endif

  // auto dev_id = static_cast<int>(ts[0].device().index());
  // // 判断是否是重物化的起点
  // bool is_threshold = remat_threshold[dev_id];
  // if(is_threshold) {
  //   // 如果是，设置对应的remat_threshold为假，
  //   remat_threshold[dev_id] = false;
  //   // 开始记录递归深度（其实不是深度而是引发的链式递归重物化的次数）
  //   recursion_depth[dev_id].push_back(0);
  // }
  // else {
  //   // 如果不是重物化起点，对应的递归深度 ++
  //   auto& vec = recursion_depth[dev_id];
  //   recursion_depth[dev_id].back() += 1;
  // }
  // // 执行重物化方程
  // auto ret = func(ts);
  // if(is_threshold) {
  //   // 重置一下
  //   remat_threshold[dev_id] = true;
  // }

#ifdef ORIG_EVICT
  if(COST_FIRST_EVICT){
  #ifdef MINIMAL_EVICT
    #ifdef MULTI_MODE
    auto device_id = static_cast<int>(ts[0].device().index());
    auto *pm = getDTBPoolManager();
    pm->auto_evict(device_id);
    #else
    pool.auto_evict();
    #endif
  #endif
  }
#endif

#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      // if(outputs.size()==2&&output_cell->pool->memory==268435456){
      //   printf("check cell defined:%d ret[%ld] defined:%d before\n", output_cell->defined ? 1 : 0, i, ret[i].defined());
      // }
      output_cell->fill(ret[i]);
      // if(outputs.size()==2&&output_cell->pool->memory==268435456){
      //   printf("check cell defined:%d ret[%ld] defined:%d after\n", output_cell->defined ? 1 : 0, i, ret[i].defined());
      // }
#ifndef ORIGINAL_DTR
      output_cell->pool->lock_remated();
#endif
    }
  }
  ecn.reset();
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
}

/* [Deprecated] */
void Rematerializer::remat(int& cumulative_num) {
#ifdef DEBUG_MODE
  if(record_er_counts){
    remat_counts += 1;
  }
  // if(remat_counts>1e5)
  //   throw std::runtime_error("Remat progress has been trapped in dead loop");
#endif
  // NOTE: author thinks that refactor using RAII for exception safety. however, RAII is not suitable for remat
  for (const strong& s : inputs) {
    s->pool->lock();
  }
  cumulative_num++;
  Tensors ts = uncheckpoint_with_depth(inputs, cumulative_num);
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif

#ifdef MULTI_MODE
  auto device_id = static_cast<int>(ts[0].device().index());
  auto *pm = getDTBPoolManager();
#endif

#ifdef MINIMAL_EVICT_COST
  #ifdef MULTI_MODE
  pm->auto_evict(device_id, memory_cost_records[rid]);
  #else
  pool.auto_evict(memory_cost_records[rid]);
  #endif
#endif

  auto ret = func(ts);

#ifdef MINIMAL_EVICT
  #ifdef MULTI_MODE
  pm->auto_evict(device_id);
  #else
  pool.auto_evict();
  #endif
#endif

#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  remat_compute_time_ += (post - pre).count();
#endif
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (auto output_cell = outputs[i].lock()) {
      output_cell->fill(ret[i]);
#ifndef ORIGINAL_DTR
      output_cell->pool->lock_remated();
#endif
    }
  }
  ecn.reset();
  for (const strong& s : inputs) {
    s->pool->unlock();
  }
}

ecn_ptr Rematerializer::get_ecn() {
  if (!ecn) {
    ecn = ecn_ptr::make(CheckpointInfo(compute_cost));
  }
  return ecn;
}

CheckpointInfo Rematerializer::get_cpi() {
  return CheckpointInfo(ecn ? duration_t(0) : compute_cost);
}

#pragma endregion



}
}