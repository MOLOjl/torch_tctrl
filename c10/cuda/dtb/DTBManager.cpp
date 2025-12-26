#pragma once
#include <c10/cuda/dtb/DTBManager.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>

namespace c10 {
namespace dtb {

size_t current_memory(int device) {
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  return device_stat.allocated_bytes[0].current;
}

size_t reserved_memory(int device){
  auto device_stat = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
  return device_stat.reserved_bytes[0].current;
}

void DTBCheckpointPool::init(int device_count) {
  printf("[TAG] init with %d\n", device_count);
  const auto size = static_cast<int64_t>(device_dtbpool.size());
  if (size < device_count) {
    device_dtbpool.resize(device_count);
    peak_allocated_memory.resize(device_count);
    peak_reserved_memory.resize(device_count);
    if_train_mode.resize(device_count);
    if_during_backward.resize(device_count);
    locked_tids.resize(device_count);
    recursion_depth_counter.resize(device_count);
    remat_counter.resize(device_count);
    for (const auto i : c10::irange(size, device_count)) {
      device_dtbpool[i] = std::make_unique<CheckpointPool>();
      peak_allocated_memory[i] = 0;
      peak_reserved_memory[i] = 0;
      if_train_mode[i] = false;
      if_during_backward[i] = false;
      recursion_depth_counter[i] = 0;
      remat_counter[i] = 0;
    }
    
  }
}

#ifndef MEM_ORDER_ENABLE
void DTBCheckpointPool::auto_evict(int device) {                 /// TAG: multi mode evict entry with minimal eviction
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  if (pool->has_memory_budget&&if_train_mode[device]) {
    int check_counts[8] = {0};
    bool if_eviction = false;
    size_t last_mem = current_memory(device);

    ///TODO: 更新当前依赖
#ifdef DEBUG_MODE
    if(record_dependcy&&current_memory(device) > pool->memory_budget){
      for (size_t i = 0; i < pool->aps.size(); i++) {
        auto ap_strong = pool->aps[i].lock();
        if (!ap_strong.defined()||ap_strong->ecn) {
          continue;
        } else {
          if (ap_strong->evictable()) {
            ap_strong->update_dependency();
          }
        }
      }
    }
#endif

    while (current_memory(device) > pool->memory_budget) {
      if_eviction = true;
#ifdef DEBUG_MODE
      if(record_mem_addr&&!current_if_any_evicted){
        DTRLogCounts("after computation, need evict.", ++evict_counts);
        pool_cur_mem_snapshot(device);  // after once computation, and out of budget
      }
      // current_if_any_evicted = true;
#endif
      pool->evict();
      check_counts[device]++;

      if(last_mem!=current_memory(device)){
        check_counts[device] = 0;
        last_mem = current_memory(device);
      }

      if(check_counts[device]>1000){
        // throw std::runtime_error("Eviction progress has been trapped in dead loop, please check if any tensors ralated to operators(jit.fuse or custom kernel) isn't dealed with.");
        c10::cuda::CUDACachingAllocator::emptyCache();
        return;
      }
    }

#ifdef DEBUG_MODE
    if(record_mem_addr&&current_if_any_evicted){
      DTRLogCounts("after eviction.", evict_counts);
      pool_cur_mem_snapshot(device);  // after eviction
      current_if_any_evicted = false;
    }
    ///TODO: 清空依赖
    if(record_dependcy&&if_eviction){
      time_t current_time = std::chrono::system_clock::now();
      for (size_t i = 0; i < pool->aps.size(); i++) {
        auto ap_strong = pool->aps[i].lock();
        if (!ap_strong.defined()||ap_strong->ecn) {
          continue;
        } else {
          if (ap_strong->evictable()) {
            auto dep = ap_strong->get_dependency();
            // DTRLogCounts("ap dep", dep);
            DTRLogDepAndCost("ap dep", dep, ap_strong->cost(current_time));
          }
        }
      }
      DTRLogCounts("once check end", 999);
    }

    // if(if_eviction){
    //     // use_log_ = true;
    //   if(record_mem_addr && pool->exts.size()>10){
    //     /// 单独用这个，记录某次驱逐后的mem snapshot
    //     pool_cur_mem_snapshot(device);
    //     // if_rec = false;

    //     /// 单独用这个，记录每次驱逐后的内存变化
    //     // DTRLogMemAlloc(current_memory(), reserved_memory());
    //   // }
    //   }
    //   record_mem_addr = false;
    // }

#endif

  }
}
      
#else
void DTBCheckpointPool::auto_evict(int device) {                 /// TAG: multi mode evict entry with mem order
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  if (pool->has_memory_budget) {
    size_t check_counts = 0;

    bool if_eviction = false;
    bool if_clear = false;
    while (current_memory(device) > pool->memory_budget) {
      if_eviction = true;
#ifdef DEBUG_MODE
      current_if_any_evicted = true;
#endif
      check_counts++;
      if(check_counts<10)
        pool->mem_first_evict(if_clear);
      else
        pool->exec_first_evict();

#ifdef DEBUG_MODE
      // if(check_counts>1000){
      //   std::cout << "Eviction progress has been trapped in dead loop\n";
      //   throw std::runtime_error("Eviction progress has been trapped in dead loop");
      // }
#endif

    }

#ifdef DEBUG_MODE
    if(check_counts>0){
      DTRLogCounts("search counts", check_counts);
      DTRLogMemAlloc(current_memory(), reserved_memory());
    }
    // if(if_eviction){
    //   if(record_mem_addr && pool->exts.size()>10){
    //     /// 单独用这个，记录某次驱逐后的mem snapshot
    //     pool_cur_mem_snapshot(device);
    //     /// 单独用这个，记录每次驱逐后的内存变化
    //     // DTRLogMemAlloc(current_memory(), reserved_memory());
    //   // }
    //   }
    // }
    //   record_mem_addr = false;
    if(record_mem_addr&&current_if_any_evicted){
      pool_cur_mem_snapshot(device);  // before computation
    }
#endif

  }
}
#endif

bool DTBCheckpointPool::auto_evict(int device, size_t coming_bytes) {
  if(!device_id_check(device)) return false;
  init_check();
  auto pool = device_dtbpool[device].get();
  long search_time_ = 0;
  time_t pre = std::chrono::system_clock::now();
  if (pool->has_memory_budget&&if_train_mode[device]) {
    int check_counts[8] = {0};
    bool if_eviction = false;
    size_t last_mem = current_memory(device);
    while ((current_memory(device) + coming_bytes) > pool->memory_budget) {
      // printf("[CHECK APS] before this evict, aps_num:%ld, current_mem:%ld, need_size:%ld\n", pool->aps.size(), current_memory(device), coming_bytes);
      pool->evict();

      check_counts[device]++;
      if(last_mem!=current_memory(device)){
        check_counts[device] = 0;
        last_mem = current_memory(device);
        if_eviction = true;
      }
      if(check_counts[device]>1000){
        // throw std::runtime_error("Eviction progress has been trapped in dead loop, please check if any tensors ralated to operators(jit.fuse or custom kernel) isn't dealed with.");
        c10::cuda::CUDACachingAllocator::emptyCache();
        return false;
      }
    }
    #ifdef DEBUG_MODE
    if(c10::dtb::trace_evicted_tensor){
      if(if_eviction){
        time_t post = std::chrono::system_clock::now();
        search_time_ += (post - pre).count();
        printf("single search time:%ld\n", search_time_);
      }
    }
    #endif
    return if_eviction;
  }else return false;
}

void DTBCheckpointPool::force_evict(int device, int mode) {
  if(!device_id_check(device)) return;
  init_check();
  auto pool = device_dtbpool[device].get();
  pool->force_evict(mode);
}

void DTBCheckpointPool::add_ap(int device, intrusive_ptr<AliasPool>& new_ap){
  if(!device_id_check(device)) return;  // ignore cpu tensor
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  auto pool = device_dtbpool[device].get();
  pool->add(new_ap);
#ifdef MEM_FIRST_EVICT
  TORCH_INTERNAL_ASSERT(new_ap.defined());
  insert_ptr2ap(reinterpret_cast<void*>(new_ap->addr), weak_intrusive_ptr<AliasPool>(new_ap));
#endif
  // }else if(device==-1){
  //   for (const auto& pool : device_dtbpool) {
  //     pool->add(new_ap);
  //   }
  // }else{
  //   throw std::runtime_error("Invalid device was detected during ap inserting.");
  // }
}

/**
 * Currently only used in mem_ordered_aps, not for normal eviction.
 * Normal eviction will remove ap during CheckpointPool::evict()
*/
void DTBCheckpointPool::erase_ap(int device, uintptr_t addr){
  auto pool = device_dtbpool[device].get();
  pool->mem_ordered_aps.erase(addr);
}

/**
 * Proactive remat tensors started from the first locked nodes.
 * remat_depth is the remat length num, 1 means remat it's neighbors
 * 2 means remat neighbors of neighors (second layer nodes)
 * Here float is just created by historical reason, acctually it is a int
 */
void DTBCheckpointPool::proactive_remat(int device, float remat_depth, bool erase) {
  init_check();
  auto pool = device_dtbpool[device].get();
  /**
   * 下面是以保留节点为起始的恢复，但不易查询到驱逐节点
   * TODO: 递归有点耗时，改为非递归写法，或者异步发起
  */
  auto it = pool->chains.begin();
  while(it != pool->chains.end() && !(*it) -> is_locked){   // find the first locked chain
    it = pool->chains.erase(it);
  }
  if(it != pool->chains.end()) {
    for(auto& cn: (*it)->members) {
      if(auto scptc = cn->value.lock()) {
        scptc->remat_neghibors(static_cast<int>(remat_depth));
      }
    }
  }
#ifdef DCR_MANAGE

#endif
  
  // pool->remat_front_batch(remat_depth, erase);
}

/**
 [deprecated]
 * record evicted tensors in eviction strageties.
  Invalid for multi batch situation because of cptc with no batch info
 */
void DTBCheckpointPool::record_evicted_tensors(int device, const weak& wcptc) {
  auto pool = device_dtbpool[device].get();
  pool->add_evited_tensor(wcptc);
}

/**
  [deprecated]
  Invalid for multi batch situation because of cptc with no batch info
 */
void DTBCheckpointPool::push_batch_evicted_tensors(int device) {
  if(device_dtbpool.empty()) return;
  auto pool = device_dtbpool[device].get();
#ifdef DEBUG_MODE
  // size_t total_mem = 0;
  // for(const auto& wcptc: pool->cur_batch_evicted_tensors){
  //   if(auto scptc = wcptc.lock()){
  //     total_mem += scptc->pool->memory;
  //     std::string rec = "device:" + std::to_string(scptc->pool->device_id) + " mem:" + std::to_string(scptc->pool->memory/1024/1024) + " MB\n";
  //     std::cout<< rec;
  //   }
  // }
#endif
  auto inserted = pool->push_single_batch_ets();
#ifdef DEBUG_MODE
  // if(inserted){
  //   std::cout << int(c10::cuda::current_device()) << " evicted " << pool->evicted_batch_tensors.back().size() 
  //     << " tensors (" << pool->evicted_batch_tensors.size() << " batch) "
  //     << "total mem:" << total_mem/1024/1024 << "MB\n";
  // }
#endif
}

/**
  [deprecated]
  Invalid for multi batch situation because of cptc with no batch info
 */
void DTBCheckpointPool::clear_recorded_batch(int device) {
  if(device_dtbpool.empty()) return;
  auto pool = device_dtbpool[device].get();
  pool->clear_recorded_batch();
}

#ifdef MEM_FIRST_EVICT
/**
 * Update ap's p2ap info.
 * Remove old_ptr records and replace it with new_addr, both of them are in the same ap.
*/
void DTBCheckpointPool::update_ap(intrusive_ptr<AliasPool>& sap, uintptr_t new_addr){
  TORCH_INTERNAL_ASSERT(sap.defined());
  auto old_addr = sap->addr;
  if(old_addr!=0){
    // sap maybe not in record.
    // auto sap = get_ap_by_ptr(reinterpret_cast<void*>(old_addr));
    // TORCH_INTERNAL_ASSERT(sap.defined(), "ptr-" + std::to_string(old_addr) + " meet a released ap occured when updating aliaspool.");
    erase_ptr2ap(reinterpret_cast<void*>(old_addr));
  }
  sap->set_addr(new_addr);
  insert_ptr2ap(reinterpret_cast<void*>(new_addr), weak_intrusive_ptr(sap));

}

/**
 * Remove addr's p2ap record.
*/
void DTBCheckpointPool::remove_p2ap(uintptr_t addr){
  // printf("[CHECK REMOVE] ptr:%ld, total:%ld\n", addr, p2ap.size());
  if(addr>0)
    erase_ptr2ap(reinterpret_cast<void*>(addr));
}

/**
 * Return finded ap of ptr, otherwise a null weak_intrusive_ptr.
*/
std::pair<weak_intrusive_ptr<AliasPool>, bool> DTBCheckpointPool::get_ap_by_ptr(void* ptr){
  auto it = p2ap.find(ptr);
  // printf("[FIND p2ap] %ld exist:%d\n", reinterpret_cast<uintptr_t>(ptr), it!=p2ap.end() ? 1 : 0);
  // TORCH_INTERNAL_ASSERT(it!=p2ap.end());
  if(it!=p2ap.end())
    return {it->second, true};
  else return {weak_intrusive_ptr<AliasPool>(intrusive_ptr<AliasPool>{}), false};
}


bool DTBCheckpointPool::check_ptr_in_aps(int device, uintptr_t addr) {
  auto pool = device_dtbpool[device].get();
  for(auto& wap: pool->aps){
    if(auto sap = wap.lock()){
      if(sap->addr==addr) return true;
    }
  }
  return false;
}

size_t DTBCheckpointPool::get_aps_size(int device) {
  auto pool = device_dtbpool[device].get();
  return pool->aps.size();
}

#endif

void DTBCheckpointPool::add_ext(int device, const weak_intrusive_ptr<External>& new_ext) {
  if(!device_id_check(device)) return;
  init_check();
#ifdef DEBUG_MODE
  update_max_meminfo(device);
#endif
  // if(likely(device>=0)){
  auto pool = device_dtbpool[device].get();
  pool->exts.push_back(new_ext);
  // }else if(device==-1){
  //   for (const auto& pool : device_dtbpool) {
  //     pool->exts.push_back(new_ext);
  //   }
  // }else{
  //   throw std::runtime_error("Invalid device was detected during exts inserting.");
  // }
}

void DTBCheckpointPool::lock_temp_ext(int device, const weak& w_cptc) {
  init_check();
  auto pool = device_dtbpool[device].get();
  pool->temp_cptc.push_back(w_cptc);
  if(auto scptc = w_cptc.lock()){       // lock temp cptc
    scptc->pool->lock();
    // DTRLogAddress("lock "+scptc->counter_name()+ " " + std::string(scptc->dtype().name()) + " device:" + std::to_string(device), 
    //       scptc->pool->addr, scptc->pool->lock_count);
    scptc->pool->if_temp = true;
  }
}


void DTBCheckpointPool::add_into_keychain(int device, const weak& new_key, const weak& pre) {
  if(!device_id_check(device)) return;
  init_check();
  auto pool = device_dtbpool[device].get();
#ifdef DEBUG_MODE
  pool->candidates.push_back(new_key);
#endif

  auto pre_node = StrongChainNode::make(pre);
  auto new_node = StrongChainNode::make(new_key);
  new_node->parent = pre_node;
  if(pool->chains.empty()){
    auto new_chain = ResidualChainRef::make(pre_node);
    new_chain->insert(new_node);
    pool->chains.push_back(new_chain);
  }else{
    bool is_find = false;
    for(const auto& chain: pool->chains){
      if(chain->in_chain(pre_node)){
        chain->insert(new_node);
        is_find = true;
        break;
      }
    }
    if(!is_find){
      auto new_chain = ResidualChainRef::make(pre_node);
      new_chain->insert(new_node);
      pool->chains.push_back(new_chain);
    }
  }

#ifdef DEBUG_MODE
  if(record_key_chain){
    if(pool->candidates.size()>0){
      auto chain_num = pool->chains.size();
      auto mem_num = pool->chains.back()->size();
      if(auto first_weak = pool->chains.back()->members[0]->value.lock())
        auto name = first_weak->counter_name();
      for(const auto& wc: pool->candidates){
        if(auto cell = wc.lock()){
          // DTRLogTensorInfo(cell->counter_name(), cell->pool->addr, cell->pool->memory, cell->get_degree(), 0, 0);
        }
      }
    }
    // record_key_chain = false;
  }
#endif
}

#ifdef DCR_MANAGE

void DTBCheckpointPool::insert_dcm(int device, nid_t s, nid_t e, const weak& s_cell, const weak& e_cell, float w) {
  init_check();
  auto pool = device_dtbpool[device].get();

  if(!pool->tmp_dcm.defined()) {
    auto new_dcm = StrongDCM::make(DCR_INIT_SIZE, DCR_INTERVAL, DCR_NB_PASS, MIN_MODULARITY, DCR_TYPE);
    pool->tmp_dcm = new_dcm;
  }
  pool->tmp_dcm->insert_single_edge(s, e, s_cell, e_cell, w);
}

void DTBCheckpointPool::add_dcm_into_queue(int device) {
  init_check();
  auto pool = device_dtbpool[device].get();
  if(!pool->tmp_dcm.defined()) return;

  pool->dcms.push_back(pool->tmp_dcm);
  pool->tmp_dcm.reset();
  
}

#endif


void DTBCheckpointPool::toggle_sampling(bool if_sampling){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_sample_tensors(if_sampling);
    }
  }
}

void DTBCheckpointPool::toggle_ignore_small_tensors(bool if_ignore){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_ignore_small_tensors(if_ignore);
    }
  }
}

void DTBCheckpointPool::set_memory_budget(long budget){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->set_memory_budget(budget);
    }
  }
}

void DTBCheckpointPool::unset_memory_budget(){
  for (const auto& pool : device_dtbpool) {
    if (pool) {
      pool->unset_memory_budget();
    }
  }
}

void DTBCheckpointPool::set_train_mode(bool flag){
  for (int i=0; i<if_train_mode.size(); i++) {
    if_train_mode[i] = flag;
  }
}

void DTBCheckpointPool::set_during_backward(bool flag){
  for (int i=0; i<if_during_backward.size(); i++) {
    if_during_backward[i] = flag;
  }
}

void DTBCheckpointPool::load_fix_tids(std::string file_path) {
  std::string log = "loaded locked_tids:";
  
  for(int i=0; i<locked_tids.size(); i++) {
    auto path_ = file_path + std::to_string(i) + ".txt";
  
    std::ifstream file(path_);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + path_);
    }  

    std::string token;
    char c;
    // 逐字符读取，支持：空格 / 逗号 / 换行 / tab
    while (file.get(c)) {
      if (std::isdigit(c) || c == '-' || c == '+') {
        token.push_back(c);
      } else {
        // 遇到分隔符，尝试解析已有 token
        if (!token.empty()) {
          try {
            long v = std::stol(token);
            locked_tids[i].insert(v);
            if(locked_tids[i].size()>300)
              break;
          } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid integer token in file " + path_ +
                ": \"" + token + "\"");
          }
          token.clear();
        }
      }
    }

    // 处理文件末尾没有分隔符的情况
    if (!token.empty()) {
      try {
        long v = std::stol(token);
        locked_tids[i].insert(v);
      } catch (const std::exception& e) {
        throw std::runtime_error(
            "Invalid integer token in file " + path_ +
            ": \"" + token + "\"");
      }
    }    
    file.close();
    log += std::to_string(locked_tids[i].size()) + "-";
  }
  // std::cout << log << "\n";
}

void DTBCheckpointPool::may_be_insert_locked(int device, const strong& cell) {
  auto tid = cell->id;
  if(locked_tids[device].find(tid) != locked_tids[device].end())
  {
    if(device_dtbpool.empty()) return;
    auto pool = device_dtbpool[device].get();
    auto wc = weak(cell);
    pool->locked_cells.push_back(wc);
    // printf("locked_cells size:%ld\n", pool->locked_cells.size());
    cell->pool->lock();
    cell->pool->is_retain = true;
  }
}

void DTBCheckpointPool::release_locked(int device) {
  if(device_dtbpool.empty()) return;
  auto pool = device_dtbpool[device].get();
  for(auto & wcell: pool->locked_cells) {
    if(auto scell = wcell.lock()) {
      scell->pool->is_retain = false;
      scell->pool->unlock();
    }
  }
}

void DTBCheckpointPool::clear_checkpointpool(int device, bool last_iter){
  if(device_dtbpool.empty()) return;          // exec without dtbpool  
  auto pool = device_dtbpool[device].get();
  if(pool->has_memory_budget){
    /// for debug, clear_exts will 
  #ifdef DEBUG_MODE
    // for (size_t i = 0; i < pool->aps.size(); i++) {
    //   auto ap_strong = pool->aps[i].lock();
    //   if (!ap_strong.defined()||ap_strong->ecn) {
    //     continue;
    //   } else {
    //     if (ap_strong->is_retain) {
    //       DTRLogCounts("ap tensors size", ap_strong->tensors.size());
    //       auto t = ap_strong->tensors.back().lock();
    //       DTRLogTensorInfo(t->counter_name(), ap_strong->addr, ap_strong->memory, 0, 0, 0);
    //       // for(const auto&t: ap_strong->tensors){
    //       //   auto cell = t.lock();
    //       //   if(cell->defined)
    //       //     DTRLogTensorInfo(cell->counter_name(), ap_strong->addr, ap_strong->memory, cell->get_degree(), 0, 0);
    //       // }
    //       ap_strong->unlock();
    //     }
    //   }
    // }
  #endif
    pool->clear_exts(last_iter);
    release_locked(device);
  }
  bwd_count ++;
  recursion_depth_total += recursion_depth_counter[device];
  recursion_depth_counter[device]=0;
  for (int i=0; i<if_during_backward.size(); i++) {
    if(i!=device)
      assert(recursion_depth_counter[i]==0);
  }
}

void DTBCheckpointPool::pool_cur_mem_snapshot(int device){
//   time_t current_time = std::chrono::system_clock::now();
//   for(const auto& ex: device_dtbpool[device].get()->exts){ 
//     if(auto ref = ex.lock()){
//       if(ref->value->defined){
//         auto& remat = ref->value->remat;
//         size_t degree = 0;
//         double cost = 0;
//         size_t staleness = 0;
//         if(remat!=nullptr){ /// TODO: 存在没有remat的tensor 无源之水，从内存大小来看是一些用到的常量直接checkpoint
//           degree = (remat->inputs.size()) + (remat->outputs.size());
//           auto ap_strong = ref->value->pool; /// TODO: 这里会触发段错误，访问了野指针？
//           cost = ap_strong->cost(current_time);
//           staleness = (current_time-ap_strong->last_used_time).count();
//         }
//         // if(ref->value->pool->memory!=0)
// #ifdef DEBUG_MODE
//         DTRLogTensorInfo(ref->value->counter_name(), ref->value->pool->addr, ref->value->pool->memory, degree, cost, staleness);
// #endif
//       }
//     }
//   }
  auto pool = device_dtbpool[device].get();
  pool->show_exts();
}


std::vector<std::pair<size_t, size_t>> DTBCheckpointPool::get_peak_memory(){
  std::vector<std::pair<size_t, size_t>> res;
  for (int i = 0; i<device_dtbpool.size(); i++) {
      res.push_back({peak_allocated_memory[i], peak_reserved_memory[i]});
  }
  clear_meminfo();
  return res;
}

/**
 * if dtb_pool is stated like `extern DTBCheckpointPool dtb_pool;` in DTBManager.h,
 * it will create multiple obj and init_once will do not work.
 * Here keep it's unique obj.
*/
DTBCheckpointPool dtb_pool;

struct BackendStaticInitializer{
  DTBCheckpointPool* parseEnvForBackend() {
    /// TODO: TO BE ADD STATIC ENV SETTING
    /* 
    const char* val = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "cudaMallocAsync")
              return CudaMallocAsync::allocator();
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    */
    return &dtb_pool;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    PoolManager.store(r);
  }
};

std::atomic<DTBCheckpointPool*> PoolManager;
BackendStaticInitializer backend_static_initializer;


void init(int device_count) {
  return get()->init(device_count);
}

void dtbPoolInitEntry() {
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  c10::dtb::init(num_devices);
#ifdef DEBUG_MODE
  signal(SIGSEGV, signal_handler);
#endif
}

static c10::once_flag dtb_init;

void lazyInitDTB() {
  c10::call_once(dtb_init, [&] { dtbPoolInitEntry(); });
}

//use pool with call: auto* poolManager = getDTBPoolManager();
DTBCheckpointPool* getDTBPoolManager() {
  return c10::dtb::get();
}
  
}
}

