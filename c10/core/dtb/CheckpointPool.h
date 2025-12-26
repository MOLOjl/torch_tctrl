#pragma once

#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/AliasPool.h>
#include <c10/core/dtb/External.h>
#include <c10/core/dtb/ResidualChain.h>
#include <c10/core/dtb/DynamicGraph.h>
#include <c10/core/dtb/DynamicClusterManager.h>


namespace c10 {
namespace dtb {

// CheckpointPool keep a list of AliasPool, and search over them to choose the best one to evict.
struct CheckpointPool {
  std::vector<weak_intrusive_ptr<AliasPool>> aps;
  std::map<uintptr_t, weak_intrusive_ptr<AliasPool>> mem_ordered_aps;   // [deprecated]

  std::vector<weak_intrusive_ptr<External>> exts;
  std::vector<weak> temp_cptc;            // during forward&backward, mark those input tensors is created casually
  std::vector<weak> candidates;           // candidates for end point      [deprecated]
  std::vector<weak> locked_cells; // [dev] record fix tensors
  std::vector<ResidualChainRef> chains; 
  StrongDCM tmp_dcm;
  std::vector<StrongDCM> dcms;            // for dcr
  std::vector<weak> cur_batch_evicted_tensors;
  std::vector<std::vector<weak>> evicted_batch_tensors; // record private evicted tensors


  std::random_device rd;
  std::mt19937 gen = std::mt19937(rd());
  // whether to take a square-root sample of the pool during an eviction loop
  bool sample_tensors = true;
  // ignore tensors < 1% of the average tensor size
  bool ignore_small_tensors = true;
  bool has_memory_budget = false;
  long memory_budget;
  void evict();
  void evict(size_t need_to_free_bytes);
  void mem_first_evict(bool&);
  void exec_first_evict();
  
  void auto_evict();
  void auto_evict(size_t size_bytes);
  /// for early check and evict
  /// for initiative evict
  void force_evict(int mode);
  void initiative_evict(size_t to_free_bytes);
  
  void add(const intrusive_ptr<AliasPool>&);
  void add_evited_tensor(const weak& wcptc);
  bool push_single_batch_ets();
  void clear_recorded_batch();
  void remat_front_batch(float scale=0.5, bool erase=true);
  void clear_dcr_records();
  // void add_chain(const intrusive_ptr<KeyChain>&);
  // void erase_chain(intrusive_ptr<KeyChain>&);
  CheckpointPool();

  /// functional
  void clear_checkpointpool();
  void set_sample_tensors(bool flag){
    sample_tensors = flag;
  }
  void set_ignore_small_tensors(bool flag){
    ignore_small_tensors = flag;
  }
  void set_memory_budget(long budget){
    memory_budget = budget;
    has_memory_budget = true;
  }
  void unset_memory_budget(){
    has_memory_budget = false;
  }
  void clear_exts(bool last_iter=true);
  void show_exts();

};

// extern CheckpointPool pool;  // cannot be extern
// extern std::unordered_map<int64_t, duration_t> compute_cost_records;
// extern std::unordered_map<int64_t, size_t> memory_cost_records;

}
}