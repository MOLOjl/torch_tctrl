#include <c10/core/dtb/CheckpointPool.h>
#include <c10/core/dtb/utils.h>
#include <c10/cuda/dtb/DTBManager.h>

#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

namespace c10 {
namespace dtb {

/**
 * Methods of CheckpointPool
 * Implementation about eviction stragety
 * Management about AliasPools in the pool
*/
#pragma region CheckpointPool

// CheckpointPool pool;
// std::unordered_map<int64_t, duration_t> compute_cost_records;
// std::unordered_map<int64_t, size_t> memory_cost_records;

/**
 * Only used in original dtr without poolManager.
 * 
*/
void CheckpointPool::add(const intrusive_ptr<AliasPool>& p) {
  // ignore storage smaller than 1% average size
  if (p->memory > 0 && (memory_count == 0 || !ignore_small_tensors || p->memory >= 0.01 * double(memory_sum/memory_count))) {
    auto new_ap = weak_intrusive_ptr<AliasPool>(p);
    aps.push_back(new_ap);

#ifdef MEM_ORDER_ENABLE
    auto result = mem_ordered_aps.insert(std::make_pair(p->addr, new_ap));
    if (!result.second) {
        // 键已存在，更新其值
        result.first->second = new_ap;
    }
#endif
  }
}

void CheckpointPool::add_evited_tensor(const weak& wcptc) {
  if(!c10::dtb::during_backward)  // only record activation evicted during forward
    cur_batch_evicted_tensors.emplace_back(wcptc);
  // if(last_flag!=c10::dtb::during_backward){
  //   if(!last_flag&&c10::dtb::during_backward) {// backward begin
  //     evicted_tensors.emplace_back(cur_batch_evicted_tensors);
  //     cur_batch_evicted_tensors.clear();
  //   }
  //   else {  // backward end

  //   }
  //   last_flag = c10::dtb::during_backward;
  // }
}

bool CheckpointPool::push_single_batch_ets() {
  bool inserted = false;
  if(!cur_batch_evicted_tensors.empty()) {
    evicted_batch_tensors.emplace_back(cur_batch_evicted_tensors);
    inserted = true;
  }
  cur_batch_evicted_tensors.clear();
  return inserted;
}

void CheckpointPool::clear_recorded_batch() {
  cur_batch_evicted_tensors.clear();
  evicted_batch_tensors.clear();
}

void CheckpointPool::remat_front_batch(float scale, bool erase) {
  auto fit = evicted_batch_tensors.begin();
  auto gap_mem = c10::dtb::memory_budget - c10::dtb::current_memory(c10::cuda::current_device());
  size_t remated_mem = 0;
  if(fit!=evicted_batch_tensors.end()) {
    // for(const auto& wcptc: *fit){
    //   if(auto scptc = wcptc.lock()) {
    //     if(scptc->pool->external_count>0){
    //       scptc->get();
    //       remated_mem += scptc->pool->memory;
    //     }
    //   }
    //   if(remated_mem>(0.5 * gap_mem)) break;
    // }
    size_t front_batch_size = fit->size();
    while(!fit->empty()) {
      auto cptcit = fit->back();
      if(auto scptc = cptcit.lock()){
        scptc->get();
        remated_mem += scptc->pool->memory;
      }
      fit->pop_back();
      if(fit->size() < (1-scale)*front_batch_size) {
        break;
      }
    }
    if(erase)
      fit = evicted_batch_tensors.erase(fit);
  }
}

#ifdef DEBUG_MODE
void log_cur_mem_statics(){ /// single mode use
#ifdef MULTI_MODE
    auto *pm = getDTBPoolManager();
    pm->pool_cur_mem_snapshot(0);
#else
    time_t current_time = std::chrono::system_clock::now();
    for(const auto& ex: pool.exts){
      if(auto ref = ex.lock()){
        if(ref->value->defined){
          auto& remat = ref->value->remat;
          size_t degree = 0;
          double cost = 0;
          if(remat!=nullptr){ /// TODO: 存在没有remat的tensor 无源之水，从内存大小来看是一些用到的常量直接checkpoint了，甚至有的常量读进来不用?
            degree = (remat->inputs.size()) + (remat->outputs.size());
            // auto ap_strong = ref->value->pool.get(); /// TODO: 这里会触发段错误，访问了野指针？
            // cost = ap_strong->cost(current_time);
          }
          // if(ref->value->pool->memory!=0)
          DTRLogTensorInfo(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory, degree, cost, 0);
        }
      }
        // DTRLogAddress(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory);
    }
    // DTRLogMemAlloc(current_memory(), reserved_memory());
#endif
}
#endif

/**
 * Only used in original dtr without poolManager
 * now use auto_evict of pm
*/
void CheckpointPool::auto_evict() {
  STATS.track("CheckpointPool::auto_evict");
  constexpr float evict_mem_scale = 0.05;
  const size_t to_free_bytes = static_cast<size_t>(memory_budget * evict_mem_scale);
  if (has_memory_budget) {
    // if(current_memory() > memory_budget * (1-evict_mem_scale)){    /// TODO: 循环释放卡死
    //   std::cout<<"initiative trigger: " << current_memory();
    //   initiative_evict(to_free_bytes);
    //   std::cout<<" | " << current_memory() << "\n";
    // }
    // 使用cuda获取只能获取到reserved的情况，而pytorch存在自己的显存池，释放只是allocated部分发生了变化
    // 因此必须使用torch的CUDACachingAllocator获取显存情况
    int check_counts = 0;
    while (current_memory() > memory_budget) {
      // std::cout<<"passive trigger\n";
      evict();
      // evict(static_cast<size_t>(evict_mem_scale * memory_budget));
      // force_evict(0);
      check_counts++;
      if(check_counts>100){
        std::cout << "Eviction progress has been trapped in dead loop\n";
        throw std::runtime_error("Eviction progress has been trapped in dead loop");
      }
    }
#ifdef DEBUG_MODE
      // use_log_ = true;
      if(record_mem_addr){
        /// 单独用这个，记录某次驱逐后的mem snapshot
        log_cur_mem_statics();
        record_mem_addr = false;
        /// 单独用这个，记录每次驱逐后的内存变化
        // DTRLogMemAlloc(current_memory(), reserved_memory());
      // }
      }
#endif
  }
}

// bool peak_out = false;
/**
 * Only used in original dtr without poolManager
*/
void CheckpointPool::auto_evict(size_t size_bytes){
  STATS.track("CheckpointPool::auto_evict");
    if (has_memory_budget) {
    // 使用cuda获取只能获取到reserved的情况，而pytorch存在自己的显存池，释放只是allocated部分发生了变化
    // 因此必须使用torch的CUDACachingAllocator获取显存情况
    // auto peak = reserved_memory();
    // if(peak>memory_budget)
    //   peak_out = true;
    // //   throw std::runtime_error("Peak memory out.");
    // if(!peak_out){
    //   while ((reserved_memory() + size_bytes) > memory_budget) {
    //     evict();
    //   }
    // }else{
    while ((current_memory() + size_bytes) > memory_budget) {
      evict();
    }
    // }
  }
}

/**
 * Only used in original dtr without poolManager
 * for debug
 * [Deprecated]
*/
void CheckpointPool::force_evict(int mode){
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  // auto evict_from_idx = [&](size_t idx) {
  //                           auto ap_strong = aps[idx].lock();
  //                           TORCH_CHECK(ap_strong.defined());
  //                           ap_strong->evict(0);
  //                           remove_from_aps(idx);
  //                         };
  size_t i = 0;
  while(i<aps.size()){
    auto cannot_evict = [&]() {
                          // shrunk = true;
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();     // 此处的lock不是自定义的Lock // 这里发生过死锁
    if (!ap_strong.defined()) {
      cannot_evict();
    }
    else if (ap_strong->ecn) {    // 当aliaspool被驱逐后，会初始化其ecn，用于remat
#ifdef DEBUG_MODE
      if(record_er_counts){
        cannot_evict_counts += 1;
      }
#endif
      cannot_evict();
    }else{
#ifdef DEBUG_MODE
      if(record_er_counts){
        evict_counts += 1;
      }
#endif
      if(ap_strong->evictable()&&!ap_strong->is_retain){  // 不在保留区间内
        // evict_from_idx(i);
        TORCH_CHECK(ap_strong.defined());
        ap_strong->evict(0);
        remove_from_aps(i);
      }else{
        i++;
      }
    }
  }
}

/**
 * Only used in original dtr without poolManager
 * [Deprecated]
*/
void CheckpointPool::initiative_evict(size_t to_free_bytes){
  size_t have_freed_bytes = 0;
  constexpr size_t stride_idx = 1;
  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                            have_freed_bytes += ap_strong->memory;
                            ap_strong->evict(0);
                            remove_from_aps(idx);
                          };
  std::uniform_int_distribution<> distrib(1, 1 * std::max(1, static_cast<int>(std::sqrt(aps.size()))));
  // while(to_free_bytes > have_freed_bytes){
    for (size_t i = 0; i < aps.size();) {
      auto cannot_evict = [&]() {
                            // shrunk = true;
                            remove_from_aps(i);
                          };
      auto ap_strong = aps[i].lock();
      if (!ap_strong.defined()) {
        cannot_evict();
      }
      else if (ap_strong->ecn) {    // 当aliaspool被驱逐后，会初始化其ecn，用于remat
  #ifdef DEBUG_MODE
        if(record_er_counts){
          cannot_evict_counts += 1;
        }
  #endif
        cannot_evict();
      }
      else if (ap_strong->is_retain){
        i += stride_idx;
      }
      else{
  #ifdef DEBUG_MODE
        if(record_er_counts){
          evict_counts += 1;
        }
  #endif
        if(i%2==0){
          ap_strong->is_retain = true;
        }else{
          if (ap_strong->evictable()){
            evict_from_idx(i);
            // i--;
          }
        }
        i += stride_idx;
      }
      if(to_free_bytes <= have_freed_bytes)
        return;
    }
// }
}

void CheckpointPool::evict() {
#ifdef ORIGINAL_DTR
  time_t pre = std::chrono::system_clock::now();
#endif 
#ifdef TIMER_ENABLE
  time_t pre = std::chrono::system_clock::now();
#endif
  STATS.track("CheckpointPool::evict");
  TORCH_CHECK(aps.size() > 0);
  // shrunk: either something has been evicted or the pools have gotten smaller
  bool shrunk = false;
  int evict_idx = -1;
  double evict_cost = INFINITY;
  time_t current_time = std::chrono::system_clock::now();

  auto remove_from_aps = [&](size_t i) {
                           aps[i] = aps[aps.size() - 1];
                           aps.pop_back();
                         };
  std::uniform_int_distribution<> distrib(1, 1 * std::max(1, static_cast<int>(std::sqrt(aps.size()))));
  // sampling a random independent subset of all evictable tensors to find the cheapest tensor to evict.
  // 搜索策略，穷举搜索aps
  for (size_t i = 0; i < aps.size();) {
    auto cannot_evict = [&]() {
                          shrunk = true;
                          #ifdef DEBUG_MODE
                          if(record_er_counts){
                            cannot_evict_counts += 1;
                          }
                          #endif
                          remove_from_aps(i);
                        };
    auto ap_strong = aps[i].lock();
    if (!ap_strong.defined()) {       // check weak_intrusive_ptr if valid
      cannot_evict();
    }
    else if (ap_strong->ecn) {
      cannot_evict();
    }
    else {
      if (ap_strong->evictable()) {
        double cost = ap_strong->cost(current_time);
      #ifdef DEBUG_MODE
        // if(record_ap_cost)
        //   DTRLogApCost("check cost", cost);
      #endif
        if (cost < evict_cost) {
          evict_cost = cost;
          evict_idx = i;
      #ifdef DEPENDENCY_CHECK
          ap_strong->update_dependency();
      #endif
        }
      }

      if (sample_tensors) {
        i += distrib(gen);
      } else {
        i += 1;
      }
    }
  }
  // 执行驱逐
  if (evict_idx == -1) {
    TORCH_CHECK(shrunk);
  } else {
#ifdef DEBUG_MODE
    if(record_er_counts){
      evict_counts += 1;
      // DTRLogEvictAPSEvents(evict_counts);
    }
#endif
    auto evict_from_idx = [&](size_t idx) {
                            auto ap_strong = aps[idx].lock();
                            TORCH_CHECK(ap_strong.defined());
                          #ifdef DEBUG_MODE
                            if(c10::dtb::trace_evicted_tensor){
                              printf("evict cost:%lf mem:%ld|",evict_cost*1e9, ap_strong->memory);
                            }
                          #endif
                            ap_strong->evict(0);
                            remove_from_aps(evict_idx);
                          };
    auto ap_strong = aps[evict_idx].lock();
#ifdef DEPENDENCY_CHECK
    // ap_strong->update_dependency();
    auto cost_dep = ap_strong->get_dependency();
    if(cost_dep<dep_threshold){
      evict_from_idx(evict_idx);
    }else{
      threshold_touch_counts++;
      if(threshold_touch_counts%10==0&&dep_threshold<max_dep_threshold)
        dep_threshold *= 2;
    }
#else
    evict_from_idx(evict_idx);
#endif

#ifdef DEBUG_MODE
    if(record_ap_cost){
      // DTRLogApCost("evicted cost", evict_cost);
      DTRLogCounts("dependecy counts", ap_strong->dependency);
    }
    // if(record_mem_addr){  // TAG: mark evicted tensor for visualization
    //   for(const auto& wp: ap_strong->tensors){
    //     if(auto cp = wp.lock()){
    //       DTRLogTensorInfo(cp->counter_name(), cp->pool->addr, cp->pool->memory, 0, 0, -999);
    //     }
    //   }
    // }
#endif
  }
#ifdef ORIGINAL_DTR
  time_t post = std::chrono::system_clock::now();
  search_time_ += (post - pre).count();
#endif
#ifdef TIMER_ENABLE
  time_t post = std::chrono::system_clock::now();
  search_time_ += (post - pre).count();
#endif
}

void CheckpointPool::exec_first_evict() {
  STATS.track("CheckpointPool::exec_first_evict");
  TORCH_CHECK(aps.size() > 0);
  // shrunk: either something has been evicted or the pools have gotten smaller
  bool shrunk = false;
  double evict_cost = INFINITY;
  auto best_it = mem_ordered_aps.begin();
  time_t current_time = std::chrono::system_clock::now();


  for (auto it = mem_ordered_aps.begin(); it != mem_ordered_aps.end();) {
    auto ap_strong = it->second.lock();
    if (!ap_strong.defined()||ap_strong->ecn) {
      it = mem_ordered_aps.erase(it);
    }
    else {
      if (ap_strong->evictable()) {
        double cost = ap_strong->cost(current_time);

        if (cost < evict_cost) {
          evict_cost = cost;
          best_it = it;
        }
      }
      it++;
      // if (sample_tensors) {
      //   i += distrib(gen);
      // } else {
      //   i += 1;
      // }
    }
  }
  // 执行驱逐
  auto ap_strong = best_it->second.lock();
  if(ap_strong->evictable()){
    ap_strong->evict(0);
    mem_ordered_aps.erase(best_it->first);
  }
}

/// @brief  TODO: bug 死锁
/// @param if_cleared 
void CheckpointPool::mem_first_evict(bool &if_cleared) {
  STATS.track("CheckpointPool::mem_first_evict");
  int evict_idx = -1;
  constexpr int search_box_size = 16;  // 相邻范围

  //   #ifdef DEBUG_MODE
      //   if(record_ap_cost)
      //     DTRLogApCost("check cost", cost);
    // #endif

  time_t current_time = std::chrono::system_clock::now();
  auto best_it = mem_ordered_aps.begin();
  double min_evict_cost = INFINITY;
  auto size_map = mem_ordered_aps.size();

  std::vector<uintptr_t> to_be_erase;
  if (size_map>search_box_size){

    for (auto it = mem_ordered_aps.begin(); std::distance(it, mem_ordered_aps.end()) >= search_box_size; it++) {
        auto end_it = std::next(it, search_box_size);
        double part_cost = 0;
        for (auto jt = it; jt != end_it; ++jt) {
          auto ap_strong = jt->second.lock();
          if (!ap_strong.defined()||ap_strong->ecn){
            to_be_erase.push_back(jt->first);
          }
          else if(!ap_strong->evictable()){  // 对不可驱逐的给一个比较大的惩罚 | 直接设为无穷会牺牲驱逐可能性
            part_cost += ap_strong->cost(current_time) * 100;
            // part_cost = INFINITY;
            // break;
          }else
            part_cost += ap_strong->cost(current_time); // 调用成员函数并累加返回值
        }

        if (part_cost < min_evict_cost) {
            min_evict_cost = part_cost;
            best_it = it;
        }

        // int i = 0;
        // while(i++ < search_box_size && std::distance(it, mem_ordered_aps.end()) >= search_box_size) it++;
    }

    // 删除具有最小代价的k个元素
    auto end_it = std::next(best_it, search_box_size);
    for(auto it = best_it; it != end_it; it++){
      auto ap_strong = it->second.lock();
      if(ap_strong.defined()&&ap_strong->evictable()){
        ap_strong->evict(0);
        to_be_erase.push_back(it->first);
      }
    }
  }else{
    for (auto it = mem_ordered_aps.begin(); it != mem_ordered_aps.end(); it++) {
      auto ap_strong = it->second.lock();
      if (!ap_strong.defined()||ap_strong->ecn){
        to_be_erase.push_back(it->first);
      }else if(ap_strong->evictable()){  // 对不可驱逐的给一个比较大的惩罚
        if (ap_strong->cost(current_time) < min_evict_cost) {
            min_evict_cost = ap_strong->cost(current_time);
            best_it = it;
        }
      }
    }
    auto ap_strong = best_it->second.lock();
    if(ap_strong.defined()&&ap_strong->evictable()){
      ap_strong->evict(0);
      to_be_erase.push_back(best_it->first);
    }
  }

  for(const auto& addr: to_be_erase)
    mem_ordered_aps.erase(addr);

}

/**
 * clear_checkpointpool finally call this
 * release locked nodes ( for kinds of methods )
 */
void CheckpointPool::clear_exts(bool last_iter){
  candidates.clear();
  // DTRLogAlias("[clear exts and if last iter]", last_iter?1:0);
  if(last_iter){
    for(auto &chain: chains){
      chain->clear_members();
    }
    chains.clear();
    // clear dcm records
    for(auto &dcm: dcms) {
      dcm->clear_comms();
    }
    dcms.clear();
    
    // clear dag records
    while(!mdags.empty()) {
      mdags.front()->clear_all_graphs();
      mdags.pop();
    }
  }else{
    auto it = chains.begin();         // 1F1B, release locked nodes like a stack order
    while(it!=chains.end()&&!(*it)->is_locked){
      it = chains.erase(it);
    }
    if(it != chains.end()) {
      (*it)->clear_members();
      chains.erase(it);
    }

    auto dit = dcms.begin();         // 1F1B, release locked nodes like a stack order
    if(dit != dcms.end()) {
      (*dit)->clear_comms();
      dcms.erase(dit);
    }

    if(!mdags.empty()) {
      mdags.front()->clear_all_graphs();
      mdags.pop();
    }
  }
  
  if(last_iter){
    // clear temp_cptc  TODO: can be optimized by std::future
    while(!temp_cptc.empty()){
      if(auto sext = temp_cptc.back().lock()){
  #ifdef DEBUG_MODE
        // if(record_cpevict_recs)
          // DTRLogAddress("clear temp begin "+sext->counter_name() + " if_tmp:"+std::to_string(sext->pool->if_temp?1:0) + " " + std::to_string(sext->pool->external_count) + std::to_string(sext->pool->lock_count), 
          //   reinterpret_cast<uintptr_t>(sext->pool->addr), sext->pool->memory);
  #endif
        // while(sext->pool->lock_count!=0)
        sext->pool->unlock();
      }
      temp_cptc.pop_back(); 
    }
  }
// #define DEBUG_MODE
/**
 * TODO: 下面即使什么不做只是清空exts，仍然会清除掉pp时要留存的张量？
*/
#ifdef DEBUG_MODE
  // show_exts();
#endif

}

void CheckpointPool::clear_dcr_records() {
  for(auto& dcm: dcms) {
    dcm->clear_comms();
  }
}

static int count = 0, pool_count = 0;
static std::map<uintptr_t, int> pool_rec;
void CheckpointPool::show_exts() {
#ifdef DEBUG_MODE
  printf("[CHECK CURRENT EXTS BEGIN]\n");
  for(auto& ele: exts) {
    if (auto e = ele.lock()) {
      // e->value->pin();  /// why pin and remat?
      count++;
      // TORCH_INTERNAL_ASSERT(e->value->defined);    // Triggered, means that exist some strong without tensor stayed in memory
      if(e->value->defined){

        if(!e->value->pool->is_evicted&&e->value->pool->device_id>=0&&!e->value->pool->if_weight){
          // auto pool_ptr = reinterpret_cast<uintptr_t>((e->value->pool.get()));
          auto pool_ptr = e->value->pool->addr;
          auto it = pool_rec.find(pool_ptr);
          int pool_id = 0;
          if(it==pool_rec.end()){
            pool_rec[pool_ptr] = ++pool_count;
            pool_id = pool_count;
          }else{
            pool_id = pool_rec[pool_ptr];
          }
          std::string tid = e->value->counter_name();
          
          printf("exts size: %ld, tid: %s, size:%ld, external_count:%ld, is_weight:%d, pool_count:%d device_id:%d, have_remat:%d input_sizes:%ld output_sizes:%ld counts:%d addr:%ld\n", 
            exts.size(), tid.c_str(), e->value->pool->memory, e->value->pool->external_count, e->value->pool->if_weight ? 1 : 0, pool_id,
            e->value->pool->device_id, e->value->pool->head_remat ? 1 : 0, 
            e->value->pool->head_remat ? e->value->pool->head_remat->inputs.size() : 0,
            e->value->pool->head_remat ? e->value->pool->head_remat->outputs.size(): 0,
            count, reinterpret_cast<uintptr_t>(e->value->t->data_ptr()));
            
        }
      }

    }
    // exts.pop_back();
  }
  printf("[CHECK CURRENT EXTS END]\n");
#endif
}


CheckpointPool::CheckpointPool() {}

// void CheckpointPool::release_resources() {
//   tmp_dcm.reset();
//   dcms.clear();
// }

#pragma endregion

    
}
}