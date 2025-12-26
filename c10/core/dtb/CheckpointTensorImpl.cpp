#include <c10/core/dtb/CheckpointTensorImpl.h>
#include <c10/core/dtb/CheckpointPool.h>
#include <c10/core/dtb/utils.h>
#include <c10/cuda/dtb/DTBManager.h>

#define TORCH_CHECK(a, ...)   // replace original TORCH_CHECK  profile mode

namespace c10 {
namespace dtb{

/**
 * Methods of CheckpointTensorCell && CheckpointTensorImpl
 * Basic but important methods about tensor's functionality
 * Important implementation about raw backend calling with pre-processing and post-processing
 * Register aliaspool and merge tensor sharing with the same ap
*/
#pragma region CheckpointTensorImpl

// CheckpointPool pool; 
// std::unordered_map<int64_t, duration_t> compute_cost_records;
// std::unordered_map<int64_t, size_t> memory_cost_records;

intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                                        bool allow_tensor_metadata_change) const{
  auto impl = c10::make_intrusive<CheckpointTensorImpl>(ref);
  TensorImpl::copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/version_counter,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopy(impl->counter_name(), counter_name());
  }
#endif
  return impl;
}

// necessary in the process of autograd, use this func to copy and detach new tensor with cpti
c10::intrusive_ptr<TensorImpl> CheckpointTensorImpl::shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const{
  auto impl = c10::make_intrusive<CheckpointTensorImpl>(ref);
  TensorImpl::copy_tensor_metadata(
        /*src_impl=*/this,
        /*dest_impl=*/impl.get(),
        /*version_counter=*/std::move(version_counter),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->refresh_numel();
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopy(impl->counter_name(), counter_name());
  }
#endif
  return impl;
}


void CheckpointTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  STATS.track("CheckpointTensorCell::shallow_copy_from");
  // auto self_impl = c10::make_intrusive<CheckpointTensorImpl>(ref);
  TORCH_CHECK(impl->key_set().has(DispatchKey::CheckpointTensorId));
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(impl.get());
  TORCH_CHECK(cpti != nullptr);
  ref->value = cpti->ref->value;
  TensorImpl::copy_tensor_metadata(
        /*src_impl=*/impl.get(),
        /*dest_impl=*/this,
        /*version_counter=*/version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  // impl->refresh_numel();
  refresh_numel();
  refresh_contiguous();
  
  if(unsafeGetTensorCell()->get().defined())
    set_sizes_and_strides(unsafeGetTensorCell()->get().sizes(), unsafeGetTensorCell()->get().strides());
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogCopyFrom(counter_name(), cpti->counter_name());
  }
#endif
}

#ifdef DEBUG_MODE
long CheckpointTensorCell::counter = 0;
#endif

#ifdef DCR_MANAGE
size_t CheckpointTensorCell::pool_counter = 0;
#endif

bool is_alias(const Tensor& l, const Tensor& r) {
  return l.defined() && r.defined() && l.is_alias_of(r);
}

// return an index for alias.
// we dont care which one because they all lead to the same alias pool.
// return -1 for no alias.
// may god forgive my sin.
int get_alias(const Tensors& ts, const Tensor& t) {
  if (t.defined()) {
    for (size_t i = 0; i < ts.size(); ++i) {
      if (ts[i].defined() && t.is_alias_of(ts[i])) {
        return i;
      }
    }
  }
  return -1;
}

CheckpointTensorImpl* get_cpti(Tensor& t) {
  // auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  // if(cpti==nullptr){  
  //   t = at::native::checkpoint(t);
  //   auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  // }
  // return cpti;
  /// TODO: BUG 存在输入是在op内部创建的tensor，其并不是cpti
  // if(!t.is_checkpoint()){
  //   t = at::native::checkpoint(t);
  // }
  auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
  return cpti;
}

Ref<intrusive_ptr<External>> cell_from_tensor(Tensor& t) {
  return get_cpti(t)->ref;
}


#ifdef DEBUG_MODE
struct OperationRecord {
  int64_t rid;
  std::string name;
  duration_t time;
  size_t mem;
  std::vector<string> inputs;
  std::vector<string> outputs;
  int device;
};
#endif

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
#ifdef DEBUG_MODE
  OperationRecord rec;
#endif
};

void add_neighbor(const strong& l, const strong& r) {
  l->pool->neighbors.push_back(weak(r));
  r->pool->neighbors.push_back(weak(l));
}

void if_tensor_add_key_chain(const weak& e, c10::dtb::DTBCheckpointPool *pm, int device){
  bool self_in = false, pre_in = false, post_in = false;
  weak pre = e;
  if(auto cell = e.lock()){
    if(C10_LIKELY(cell->get_degree()!=RESIDUAL_DEGREE)) return;
    self_in = true;
    if(C10_LIKELY(cell->remat)){
      for(const auto& it: cell->remat->inputs){
        if(it->get_degree()==RESIDUAL_DEGREE){
          pre_in = true;
          pre = weak(it);
          break;
        }
      }
      for(const auto& wot: cell->remat->outputs){
        if(auto ot = wot.lock()){
          if(ot->get_degree()==RESIDUAL_DEGREE){
            post_in = true;
            break;
          }
        }
      }
    }
  }
  if(self_in&&pre_in&&post_in){
    pm->add_into_keychain(device, e, pre);
  }
}

void if_tensor_add_key_chain(const strong& cell, c10::dtb::DTBCheckpointPool *pm, int device){
  bool self_in = false, pre_in = false, post_in = false;
  weak pre = weak(cell);
  if(C10_LIKELY(cell->get_degree()!=RESIDUAL_DEGREE)) return;
  self_in = true;
  if(C10_LIKELY(cell->remat)){
    for(const auto& it: cell->remat->inputs){
      if(it->get_degree()==RESIDUAL_DEGREE){
        pre_in = true;
        pre = weak(it);
        break;
      }
    }
    for(const auto& wot: cell->remat->outputs){ /// TODO: 这里其实是判断了自己，但是不进行这个判断会漏掉
      if(auto ot = wot.lock()){
        if(ot->get_degree()==RESIDUAL_DEGREE){
          post_in = true;
          break;
        }
      }
    }
  }
  if(self_in&&pre_in&&post_in){
    pm->add_into_keychain(device, weak(cell), pre);
  }
}

inline void release_external_of_nosource_tensor(const strong& s, const std::string& name){
  if(!s->pool->if_weight && s->pool->head_remat==nullptr && during_backward && name != "copy_"){    // [BUG]: copy_ is a complex bug for DTR runtime, cannot change orginal tensor easily
    // s->pool->release_external();                         /// BUG: this way will cause segmentation fault, probably related to free tensor needed
  }
}

#ifdef ARITHMETIC_TEST
static size_t cur_op_counts = 0;
static constexpr size_t evict_num = 2;
#endif

/* remat take a single vector of tensors,
while there are two vector, one storing nonconstants and one storing constants.
the constants are small and they will not be considered for eviction.
however, we have to stitch the two vectors together to pass it in remat.
the size_t in constants decide the location to stitch them in, while input_values fill in the rest. */

MakeRawResult make_raw(const rematerialize_function_t& remat_f, const strongs& inputs, const std::string& name) {
  STATS.track("make_raw");
  #ifdef ARITHMETIC_TEST
  cur_op_counts++;
  #endif
  // bool if_res_retain = false;
  for (const strong& s : inputs) {                  // lock for unevictable
    s->pool->lock();
    // if(!s->pool->head_remat.defined()) {
    //   if_res_retain = true;
    // }
  }
#ifdef DEPTH_ENABLE
  int cumulative_num = 0;
  Tensors raw_inputs = uncheckpoint_with_depth(inputs, cumulative_num);
#else
  Tensors raw_inputs = uncheckpoint(inputs);        // cancel the monitor and get tensors
#endif
  auto device_id = static_cast<int>(raw_inputs[0].device().index());  /// do not influence device

  time_t pre = std::chrono::system_clock::now();
  auto raw_outputs = remat_f(raw_inputs);
  time_t post = std::chrono::system_clock::now();
  auto cur_compute_cost = post - pre;
  // auto cur_compute_cost = test_time_post - test_time_cur;

  auto* pm = getDTBPoolManager();
#ifdef ORIG_EVICT
  if(COST_FIRST_EVICT){
    #ifdef MULTI_MODE
    pm->auto_evict(device_id);
    #else
    pool.auto_evict();
    #endif
  }
#endif

#ifdef ORIGINAL_DTR
  base_compute_time_ += (post - pre).count();       // if sync?
  // c10::cuda::device_synchronize();
#endif
#ifdef TIMER_ENABLE
  base_compute_time_ += (post - pre).count();       // if sync?
#endif
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, cur_compute_cost);

  for (Tensor& t : raw_outputs) {             // prepare checkpoint for raw_outputs
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);           // check if t is an alias of tensor in inputs
    auto m = memory(t);
    auto addr = get_addr(t);
#ifdef DEBUG_MODE
    bool new_ap = true;
#endif
    if (alias == -1) {
      // alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, device_id);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, addr, device_id);     /// [TAG] AliasPool构造
      // if(reserved_range){     /// TAG: 保留区间内的aps保存
      //   alias_pool->is_retain = true;
      //   alias_pool->lock();    /// 一直保存了，需要主动释放
      // }
#ifdef MULTI_MODE
      pm->add_ap(device_id, alias_pool);
#else
      pool.add(alias_pool);     /// TAG: alaispool新增的唯一入口
#endif
    }
    else {
      alias_pool = inputs[alias]->pool;
#ifdef DEBUG_MODE
      if(trace_register_and_release){
        new_ap = false;
      }
#endif
#ifdef MEM_FIRST_EVICT
      pm->update_ap(alias_pool, addr);
      // alias_pool->set_addr(addr);    // TODO[√]: why org addr become a strange addr? because original ptr becomes an undefined ptr
#else
      alias_pool->set_addr(addr);    // TODO[√]: why org addr become a strange addr? because original ptr becomes an undefined ptr
#endif
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += cur_compute_cost;
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat); // bind external for t

    // pyf_debug
    // lock tensors with fix tid
    pm->may_be_insert_locked(device_id, e->value);

#ifdef DEBUG_MODE
    if(trace_register_and_release){
      printf("make_raw(%s) new external, new ap:%d, addr:%ld\n", name.c_str(), new_ap?1:0, e->value->pool->addr);
    }
#endif

#ifdef MULTI_MODE
    pm->add_ext(device_id, weak_intrusive_ptr<External>(e));
#else
    pool.exts.push_back(weak_intrusive_ptr<External>(e));
#endif
#ifdef DEGREE_CHAIN
    if(!during_backward&&pm->if_train_mode[device_id]){ /// change degree for outputs
      e->value->add_degree(inputs.size());
      // if_tensor_add_key_chain(weak(e->value), pm, device_id);  // output check cannot find residual
    }
#endif
    alias_pool->tensors.push_back(weak(e->value));                // same storage in one alias_pool
    // alias_pool->is_retain = if_res_retain;
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }

  remat->outputs = weak_outputs;
  // make each pair of tensor which are not alias relation to be neighbor
  for (size_t i = 0; i < inputs.size(); ++i) {
#ifdef DEGREE_CHAIN
    if(!during_backward&&pm->if_train_mode[device_id]){   /// change degree for inputs
      inputs[i]->add_degree(outputs.size());
      if_tensor_add_key_chain(inputs[i], pm, device_id);
    }
#endif
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        add_neighbor(inputs[i], outputs[j]->value);
      }

#ifdef DCR_MANAGE
      if(DCR_LOCK_ENABLE)
        if(!during_backward&&pm->if_train_mode[device_id]) {  // during forward
          pm->insert_dcm(device_id, inputs[i]->dg_id, outputs[j]->value->dg_id, weak(inputs[i]), weak(outputs[j]->value));  // TODO: maybe weight add?
        }
#endif

    }
  }
#ifdef DCR_MANAGE
  if(DCR_LOCK_ENABLE)
    if(during_backward&&pm->if_train_mode[device_id]) {
      // 反向且tmp dcm非空，加入dcms
      pm->add_dcm_into_queue(device_id);
    }
#endif

  for (const strong& s : inputs) {
    s->pool->unlock();
    // release_external_of_nosource_tensor(s, name);
    // #ifdef ARITHMETIC_TEST // stupid
    // if(cur_op_counts % evict_num == 0){
    //   if(s->pool->evictable())
    //   s->pool->evict(0);
    // }
    // #endif
  }

#ifdef DEBUG_MODE
  return {outputs, aliases, cur_compute_cost, remat, {{},name, cur_compute_cost, {}, {},{}, device_id}};
#else
  return {outputs, aliases, cur_compute_cost, remat};
#endif
}

template <class T>
inline void hash_combine(int64_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// 自定义的哈希函数, int64_t是IntArrayRef.vec()的默认类型
struct VectorHash {
    int64_t operator()(const std::vector<int64_t>& v) const {
        int64_t hash = 0;
        for (auto &num : v) {
            // 结合每个元素的哈希值，这里使用了位移和异或来混合哈希值  //  0×9e3779b9是 黄金分割数 乘 2^32次方 的16进制表示，32位无符号整数中它处在黄金分割的位置
            hash_combine(hash, num);
        }
        return hash;
    }
};

// 添加缓存机制来避免重复哈希字符串
std::unordered_map<std::string, int64_t> functionNameHashCache;

inline int64_t generateUniqueHash(const std::string& functionName, const Tensors &inputs) {
    int64_t hash = 0;

    // 从缓存中获取或计算函数名的哈希值
    auto it = functionNameHashCache.find(functionName);
    if (it == functionNameHashCache.end()) {
        int64_t fnHash = std::hash<std::string>{}(functionName) + 0x9e3779b9;
        functionNameHashCache[functionName] = fnHash;
        hash ^= fnHash;
    } else {
        hash ^= it->second;
    }

    // 为输入尺寸生成哈希值并合并
    VectorHash vectorHasher;
    for(auto &input: inputs){
        hash_combine(hash, vectorHasher(input.sizes().vec()));
    }

    return hash;
}


// func name and size for cost model
MakeRawResult make_raw_rec(const rematerialize_function_t& remat_f,
                       const strongs& inputs, const std::string& name) {
  STATS.track("make_raw with hash");

  for (const strong& s : inputs) {                  // lock for unevictable
    s->pool->lock();
  }
#ifdef DEPTH_ENABLE
  int cumulative_num = 0;
  Tensors raw_inputs = uncheckpoint_with_depth(inputs, cumulative_num);
#else
  Tensors raw_inputs = uncheckpoint(inputs);        // cancel the monitor and get tensors
#endif
  Tensors raw_outputs;
  auto device_id = static_cast<int>(raw_inputs[0].device().index());

  auto rid = generateUniqueHash(name, raw_inputs);
  auto search_res = compute_cost_records.find(rid);
  bool have_record = false;
  duration_t cur_compute_cost;
  size_t cur_mem_cost;
  if(search_res != compute_cost_records.end()){ // 有记录
    have_record = true;
  }

#ifdef DEPTH_ENABLE
  if(cumulative_num>0){
    DTRLogCalculativeRematsRecords(rid, name, cumulative_num);
  }
#endif

#ifdef MULTI_MODE
  auto *pm = getDTBPoolManager();
#endif
  if(have_record){
    cur_mem_cost = memory_cost_records[rid];
#ifdef ORIG_EVICT
  if(COST_FIRST_EVICT){
    #ifdef MULTI_MODE
      pm->auto_evict(device_id, cur_mem_cost);
    #else
      pool.auto_evict(cur_mem_cost);
    #endif
  }
#endif
    cur_compute_cost = compute_cost_records[rid];
    raw_outputs = remat_f(raw_inputs);
  }else{
    size_t pre_mem = current_memory();
    time_t pre_time = std::chrono::system_clock::now();
    raw_outputs = remat_f(raw_inputs);
    time_t post_time = std::chrono::system_clock::now();
    size_t post_mem = current_memory();
    cur_compute_cost = post_time - pre_time;
    // insert new record item
    compute_cost_records[rid] = cur_compute_cost;
    cur_mem_cost = post_mem - pre_mem;
    memory_cost_records[rid] = cur_mem_cost;
#ifdef ORIG_EVICT
  if(COST_FIRST_EVICT){
    #ifdef MULTI_MODE
      pm->auto_evict(device_id);
    #else
      pool.auto_evict();
    #endif
  }
#endif
  }
  
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  weaks weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, rid, cur_compute_cost);

  for (Tensor& t : raw_outputs) {             // prepare checkpoint for raw_outputs
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);           // if t is an alias of tensor in inputs?
    auto m = memory(t);
    auto addr = get_addr(t);
    if (alias == -1) {
      // alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, device_id);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, m, addr, device_id);    /// [TAG] AliasPool构造
#ifdef MULTI_MODE
      pm->add_ap(device_id, alias_pool);
#else
      pool.add(alias_pool);     /// TAG: alaispool新增的唯一入口
#endif
    }
    else {
      alias_pool = inputs[alias]->pool;
#ifdef MEM_FIRST_EVICT
      pm->update_ap(alias_pool, addr);
      // alias_pool->set_addr(addr);    // TODO[√]: why org addr become a strange addr? because original ptr becomes an undefined ptr
#else
      alias_pool->set_addr(addr);    // TODO[√]: why org addr become a strange addr? because original ptr becomes an undefined ptr
#endif
      if (alias_pool->head_remat) {
        alias_pool->head_remat->compute_cost += (cur_compute_cost);
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat); // bind external for t
#ifdef MULTI_MODE
    pm->add_ext(device_id, weak_intrusive_ptr<External>(e));
#else
    pool.exts.push_back(weak_intrusive_ptr<External>(e));
#endif
    alias_pool->tensors.push_back(weak(e->value));                // same storage in one alias_pool
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(weak(outputs.back()->value));
  }

  remat->outputs = weak_outputs;
  // make each pair of tensor which are not alias relation to be neighbor
  for (size_t i = 0; i < inputs.size(); ++i) {
    for (size_t j = 0; j < outputs.size(); ++j) {
      if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        add_neighbor(inputs[i], outputs[j]->value);
      }
    }
  }
  for (const strong& s : inputs) {
    s->pool->unlock();
    // release_external_of_nosource_tensor(s, name);
  }
#ifdef DEBUG_MODE
  OperationRecord rec = {rid, name, cur_compute_cost, cur_mem_cost, {}, {}, device_id};
  return {outputs, aliases, cur_compute_cost, remat, rec};
#else
  return {outputs, aliases, cur_compute_cost, remat};
#endif
}

std::string from_time(duration_t t) {
  return std::to_string(std::chrono::nanoseconds(t).count());
}

/**
 * 广泛使用在Checkpoint后端
*/
Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   Tensors& inputs) {
  STATS.track("CheckPointTensorImpl::make");
  Tensors checkpointed_inputs = try_checkpoint(inputs); // make intrusive_ptr for inputs
  auto input_size = checkpointed_inputs.size();

  strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    // if(cpti->unsafeGetTensorCell()->pool->external_count>1){
    //   if(record_lifecycle){ // 记录关注的tensor
    //     DTRLogLifeCycle(name, cpti->unsafeGetTensorCell()->pool->external_count, cpti->unsafeGetTensorCell()->pool->lock_count, cpti->unsafeGetTensorCell()->pool->remat_count);
    //   }
    // }
    
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }

#if defined(MINIMAL_EVICT) || defined(MEM_FIRST_EVICT)
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
  }

#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (const auto& t: tensors) {
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      output_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  // if (use_log_) {
  //   std::vector<std::string> res;
  //   res.reserve(ret.outputs.size());

  //   for (const auto& tensor : tensors) {
  //     res.push_back(get_cpti(tensor)->counter_name());
  //   }

  //   DTRLogCall(res, name, args, from_time(ret.time));
  //   for (size_t i = 0; i < tensors.size(); ++i) {
  //     Tensor t = tensors[i];
  //     auto cpti = get_cpti(t);
  //     DTRLogMemory(cpti->counter_name(), cpti->unsafeGetTensorCell()->memory());
  //     DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
  //     // DTRLogAlias(cpti->counter_name(), ret.aliases[i], cpti->unsafeGetTensorCell()->pool->tensors.size());
  //   }
  // }
#endif

  return tensors;
}

Tensors CheckpointTensorImpl::make(const std::string& name,
                                   const rematerialize_function_t& remat,
                                   Tensors&& inputs) {
  STATS.track("CheckPointTensorImpl::make");
  Tensors checkpointed_inputs = try_checkpoint(inputs); // make intrusive_ptr for inputs
  auto input_size = checkpointed_inputs.size();

  strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    // if(cpti->unsafeGetTensorCell()->pool->external_count>1){
    //   if(record_lifecycle){ // 记录关注的tensor
    //     DTRLogLifeCycle(name, cpti->unsafeGetTensorCell()->pool->external_count, cpti->unsafeGetTensorCell()->pool->lock_count, cpti->unsafeGetTensorCell()->pool->remat_count);
    //   }
    // }
    
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }

#if defined(MINIMAL_EVICT) || defined(MEM_FIRST_EVICT)
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<CheckpointTensorImpl>::make(t));
    tensors.push_back(cp);
  }

#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (const auto& t: tensors) {
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      output_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  // if (use_log_) {
  //   std::vector<std::string> res;
  //   res.reserve(ret.outputs.size());

  //   for (const auto& tensor : tensors) {
  //     res.push_back(get_cpti(tensor)->counter_name());
  //   }

  //   DTRLogCall(res, name, args, from_time(ret.time));
  //   for (size_t i = 0; i < tensors.size(); ++i) {
  //     Tensor t = tensors[i];
  //     auto cpti = get_cpti(t);
  //     DTRLogMemory(cpti->counter_name(), cpti->unsafeGetTensorCell()->memory());
  //     DTRLogAlias(cpti->counter_name(), ret.aliases[i]);
  //     // DTRLogAlias(cpti->counter_name(), ret.aliases[i], cpti->unsafeGetTensorCell()->pool->tensors.size());
  //   }
  // }
#endif

  return tensors;
}

// TODO: check that mutated value does not have alias.
void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  Tensors& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                  //  new_input_values[idx] = t[idx].clone();    /// TODO: 绕开clone
                   new_input_values[idx] = t[idx];
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  strongs input_values;
  std::vector<std::string> args;
  // 拿到tensor本体
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }
#if defined(MINIMAL_EVICT) || defined(MEM_FIRST_EVICT)
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  const auto& modified = ret.outputs;
  for (size_t idx: mutate_idx) {                              /// TODO: 可能存在inputs中不为cpti的tensor，但受限于语法无法直接修改
    // if(C10_UNLIKELY(!native::is_checkpoint(inputs[idx])))
    //   inputs[idx] = native::checkpoint(inputs[idx]);
    if(!inputs[idx].is_checkpoint()){
      inputs[idx] = at::native::checkpoint(inputs[idx]);
    }
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (size_t idx: mutate_idx) {
      output_ids.push_back(cell_from_tensor(inputs[idx])->value->value->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  if (use_log_) {
    DTRLogMutate(name, args, mutate_idx, from_time(ret.time));
  }
#endif
}

void CheckpointTensorImpl::mutate(const std::string& name,
                                  const mutate_function_t& mutate,
                                  Tensors&& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                  //  new_input_values[idx] = t[idx].clone();    /// TODO: 绕开clone
                   new_input_values[idx] = t[idx];
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  strongs input_values;
  std::vector<std::string> args;
  // 拿到tensor本体
  for (const Tensor& t: checkpointed_inputs) {
    auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
    TORCH_CHECK(cpti);
    input_values.push_back(cpti->unsafeGetTensorCell());
#ifdef DEBUG_MODE
    if (use_log_) {
      args.push_back(cpti->counter_name());
    }
#endif
  }
#if defined(MINIMAL_EVICT) || defined(MEM_FIRST_EVICT)
  auto ret = make_raw(remat, input_values, name);
#endif
#ifdef MINIMAL_EVICT_COST
  auto ret = make_raw_rec(remat, input_values, name);
#endif

  const auto& modified = ret.outputs;
  /**
   * 需要说明的是，这里不论是传入右值引用还是左值引用，都无法对原本的Tensor进行修改
   * 这里的处理只是让程序执行时不存在bug，并不能达到真正的目的——让非cpti的Tensor变为cpti
   * 在checkpoint.cpp中封装的函数，传入值存在const
  */
  for (size_t idx: mutate_idx) {                              /// TODO: 可能存在inputs中不为cpti的tensor，但受限于语法无法直接修改
    if(C10_UNLIKELY(!at::native::is_checkpoint(inputs[idx])))
      inputs[idx] = at::native::checkpoint(inputs[idx]);
    cell_from_tensor(inputs[idx])->value = modified[idx];
  }
#ifdef DEBUG_MODE
  if(record_op_recs){
    std::vector<string> input_ids;
    std::vector<string> output_ids;
    for (const auto& t: checkpointed_inputs) {  // bind External for inputs intrusive_ptr
      auto* cpti = dynamic_cast<CheckpointTensorImpl*>(t.unsafeGetTensorImpl());
      input_ids.push_back(cpti->unsafeGetTensorCell()->counter_name());
    }
    for (size_t idx: mutate_idx) {
      output_ids.push_back(cell_from_tensor(inputs[idx])->value->value->counter_name());
    }
    ret.rec.inputs = std::move(input_ids);
    ret.rec.outputs = std::move(output_ids);
    DTRLogOPRecords(ret.rec.rid, ret.rec.name, static_cast<int64_t>(std::chrono::duration_cast<std::chrono::microseconds>(ret.rec.time).count()),
                    ret.rec.mem, ret.rec.inputs, ret.rec.outputs, ret.rec.device);
  }
  if (use_log_) {
    DTRLogMutate(name, args, mutate_idx, from_time(ret.time));
  }
#endif
}

void CheckpointTensorImpl::release_resources() {
#ifdef DEBUG_MODE
  if (use_log_) {
    DTRLogRelease(counter_name());
  }
#endif
  ref.reset();
}

CheckpointTensorImpl::CheckpointTensorImpl(Tensor& t, bool if_weight) : CheckpointTensorImpl(c10::make_intrusive<External>(t, if_weight)) {
  // set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
  // set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
  if(unsafeGetTensorCell()->t->defined())
  {
    set_sizes_and_strides(unsafeGetTensorCell()->get().sizes(), unsafeGetTensorCell()->get().strides());
    /// original DTR do not add directly transfered tensor into it's pool
    /// this is for the Fine grained memory management

    /**
     * Original design do not really move content of t, so this should distinguish if outer tensor or inner tensor, otherwise double free.
     * Because that outer tensor is still have content, but when pool release the same memory.
     * The outer tensor(original t)'s deconstructor will try to free the same memory, so segmentation fault.
     * But with real moving of t, the original t does not have any content.
    */
    unsafeGetTensorCell()->pool->tensors.push_back(weak(unsafeGetTensorCell()));
  }
#ifdef MULTI_MODE
  /**
   * really std::move will clear t's content, so here will get the device_id all -1!!!
  */
  // auto device_id = C10_LIKELY(t.defined()) ? static_cast<int>(t.device().index()) : -1;      /// CPU data possible or undefiend tensor
  auto device_id = C10_LIKELY(unsafeGetTensorCell()->defined) ? static_cast<int>(unsafeGetTensorCell()->pool->device_id) : -1;      /// CPU data possible or undefiend tensor
  auto *pm = getDTBPoolManager();
  pm->add_ext(device_id, weak_intrusive_ptr<External>(ref->value));
#ifdef DEBUG_MODE
  if(trace_register_and_release){
    printf("checkpoint new external, new ap:1, deivce:%d\n", device_id);
  }
  // if(!if_weight){
  //   pm->lock_temp_ext(c10::cuda::current_device(), weak(unsafeGetTensorCell()));
  // }
  if(record_cpevict_recs) {
    DTRLogAddress("outer checkpoint "+counter_name()+ " " + std::string(unsafeGetTensorCell()->dtype().name()) + " device:" + std::to_string(device_id), 
      unsafeGetTensorCell()->pool->addr, unsafeGetTensorCell()->pool->memory);
  }
#endif
  /**
   * model weights maybe init on CPU, so cannot be add here
   * aps only records CUDA tensors
  */
  // printf("[ADD_AP before] %ld %d\n", ref->value->value->pool->addr, device_id);
  // pm->add_ap(device_id, ref->value->value->pool);                                       
#else
  pool.exts.push_back(weak_intrusive_ptr<External>(ref->value));
#endif
}

/**
 * In pytorch2.1, TensorImpl is not used with intrusive_ptr as a member of Tensor,
 * although TensorImpl is still inherited from intrusive_target_ptr.
 * So it has to use deconstructor to call reset.(Just more clear, ref will be reset even without this)
*/
CheckpointTensorImpl::~CheckpointTensorImpl() {
  // printf("cpti deconstruct trigger ");
  ref.reset();
}

#pragma endregion





}
}