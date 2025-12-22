#pragma once

// #ifdef BUILDING_COMMON
// export declaration for visibility out of the .so
#define COMMON_API __attribute__((visibility("default")))
// #else
// #define COMMON_API
// #endif

#include <atomic>
#include <memory>
#include <numeric>
#include <random>
#include <future>
#include <chrono>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <c10/core/dtb/Logger.h>
#include <cuda_runtime.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

// #define ORIGINAL_DTR
#define DEBUG_MODE

#define MINIMAL_EVICT                    /// 最小驱逐策略（贪心+随机 DTR）
// #define MINIMAL_EVICT_COST            /// 最小驱逐策略+cost cache（贪心+随机 DTR） op记录
#define DEGREE_CHAIN                     /// 残差链发现策略
#define MEM_FIRST_EVICT                  /// 以内存为中心的驱逐策略(unified_evict)
#define DCR_MANAGE                      /// 动态社区算法
// #define ORIG_EVICT                       /// DTR original Evction
// 集群上的cost_evict也使用了single_pool + pre_eviction的优化
#define PROACTIVE_REMAT                 /// 主动恢复相关接口

// #define ARITHMETIC_TEST                 /// 等差释放测试

#define MULTI_MODE                      /// 是否启用多卡管理模式，默认开启，不开启应该目前跑不了
/**
 * 为测试方便，不用重新编译，都采用环境变量来控制不同优化是否启用
 * 尽管上述开关均打开，会带来额外的指令开销
*/

// #define TIME_REC                      /// [deprecated]方便debug的宏定义
// #define MEM_ORDER_ENABLE              /// [deprecated]是否启用mem order策略
// #define DEPENDENCY_CHECK              /// [deprecated]依赖检查策略

// #define TIMER_ENABLE                 /// [deprecated] 是否启用计时(debug)
// #define DEPTH_ENABLE                 /// [deprecated] 记录每次重物化所累计恢复的张量个数

/* 控制变量区 */

static const int RESIDUAL_DEGREE = ([]() -> int {    /// 残差链度设置  4-Llama2-7b-hf 6-GPT_simp
    const char* env = getenv("RESIDUAL_DEGREE");
    if(env) return atoi(env);
    else return 99;
})();
// constexpr const int RESIDUAL_DEGREE = 6;  /// 残差链度设置  4-Llama2-7b-hf 6-GPT_simp

static const bool COST_FIRST_EVICT = ([]() -> bool {
    const char* env = getenv("COST_FIRST_EVICT");
    if(env) return (atoi(env))==1;
    else    return false;
})();

static const bool UNIFIED_EVICT = !COST_FIRST_EVICT;

constexpr const int dep_threshold = 50;             /// 重物化链深度阈值
static int threshold_touch_counts = 0;     /// 累积触发次数
constexpr const int max_dep_threshold = 500;

#define WEIGHTED   1
#define UNWEIGHTED 0

static const bool DCR_LOCK_ENABLE = ([]() -> bool {
    const char* env = getenv("DCR_LOCK_ENABLE");
    if(env) return (atoi(env))==1;
    else    return false;
})();

static const int DCR_INIT_SIZE = ([]() -> int {  // nb_nodes > cluster_init_size, then initial com
    const char* env = getenv("DCR_INIT_SIZE");
    if(env) return atoi(env);
    else return 1000;
})();      
static const int DCR_INTERVAL  = ([]() -> int { // △nb_nodes > cluster_interval, then dynamic change com
    const char* env = getenv("DCR_INTERVAL_SIZE");
    if(env) return atoi(env);
    else return 200;
})();      
static const int DCR_LOCK_TOPS  = ([]() -> int { // △nb_nodes > cluster_interval, then dynamic change com
    const char* env = getenv("DCR_LOCK_TOPS");
    if(env) return atoi(env);
    else return 4;
})();
constexpr const int DCR_NB_PASS = -1;                  // for com
constexpr const double MIN_MODULARITY = 0.000001;      // for com
constexpr const int DCR_TYPE = UNWEIGHTED;


#ifdef TIME_REC
auto start_time = std::chrono::high_resolution_clock::now();
auto end_time = std::chrono::high_resolution_clock::now();                                    
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
#endif
#ifdef TIME_REC
  #define START_TIMER start_time = std::chrono::high_resolution_clock::now();
#else
  #define START_TIMER
#endif
#ifdef TIME_REC
  #define END_TIMER(tag){                                                                       \
  end_time = std::chrono::high_resolution_clock::now();                                    \
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); \
  std::cout << (tag) << duration.count() << " ms" << std::endl;                                 \
  }
#else
  #define END_TIMER(tag) {auto r = {tag};}
#endif

// System Description:
// Every Tensor is managed by a CheckpointTensor,
// that describe how it is computed, (the function and the inputs)
// And might optionally hold the tensor value.
// The tensor value might be dropped, and when requested later, recomputed and cached again.

// Corner Cases:
// A CheckpointedTensor might require_grad.
//   In this case the underlying data must not require_grad,
//   as we want backpropagation on the outer, uncheckpoined level.
//   To be more specific, suppose a tensor is recomputed multiple times.
//   We want to only compute the gradient exactly once.
//   To do this, the wrapper must be require_grad, and the wrapped value must not.
// A CheckpointedTensor might be constant.
//   In this case it is unevictable.
// An operator might return multiple output.
//   In this case the computation info (rematerializer) is shared between all of them,
//   And when the function get computed again all value get cached.
// An operator might not return value, but only mutate input value.
//   To combat this, we COW the operator, and wrap CheckpopintTensor with a Ref.
//   By doing this the inner CheckpointTensor is kept purely functional.
// An operator might try to mutate uncheckpointed tensor.
//   We do not support this and will error.
// An operator might create aliases.
//   We track alias in AliasPool.
//   Each AliasPool hold a set of tensor that is alias to eachother.
// An operator might try to create Alias to an unevictable tensor.
//   In such a case the output tensor is unevictable.
// An operator might try to mutate Tensor with Alias.
//   We do not support this case an will error if a Tensor has any alive Alias.
//   However it could be done without a major redesign of the system -
//   Each AliasPool will hold weak pointers to the External Reference.
//   When alias mutation occur,
//   we make a rematerialize_function that take in the base tensor (other tensor alias from)
//   and output all the new value of the aliases, then update the Ref.
//   Of course, the cleaner way is to not support this.
//   Shame on those who use this feature.

// Memory Safety:
// The objects here will have lots of backedges.
// In order to collect memory when computation is completed,
// We require that all strong pointer is of the form of value -> input.
// This ensure that everything will be released if there is no external ref whatsoever.

// Optimization:
// We treat tensor that has no external reference differently -
// They will never be externally used again so we assume their next use time is infinite
// so, if it doesnt has any evicted neighbor it will get evicted immediately.

// Note: to code fast I do not use RAII and just assume the code will not try to recover from exception.
// It should be easy to fix though.

namespace at
{
  class Tensor;
  extern COMMON_API bool reserved_range;
  extern COMMON_API bool during_backward;
}


namespace c10{
namespace dtb{

using at::Tensor;

class CheckpointTensorCell;
using strong = intrusive_ptr<CheckpointTensorCell>;
using strongs = std::vector<strong>;
using weak = weak_intrusive_ptr<CheckpointTensorCell>;
using weaks = std::vector<weak>; 
using Tensors = std::vector<Tensor>;
using rematerialize_function_t = std::function<Tensors(const Tensors&)>;
using mutate_function_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Duration = Clock::duration;
using FinalTime = std::chrono::nanoseconds;

using at::reserved_range;
using at::during_backward;

// assignment is in aten/src/ATen/CheckpointTensorImpl.cpp
extern size_t memory_sum;
extern size_t memory_max;
extern size_t memory_count;

// constexpr const int CHAIN_LENGTH_LOCK_THRESHOLD = 8;  // 16
// constexpr const int CHAIN_LOCK_STRIDE = 2;            // llama2 lora use 2 for faster remat, megatron-lm use 4 for test, maybe can use 2 too?
static const int CHAIN_LENGTH_LOCK_THRESHOLD = ([]() -> int {
    const char* env = getenv("CHAIN_LENGTH_LOCK_THRESHOLD");
    if(env) return (atoi(env));
    else    return 8;
})();
static const int CHAIN_LOCK_STRIDE = ([]() -> int {
    const char* env = getenv("CHAIN_LOCK_STRIDE");
    if(env) return (atoi(env));
    else    return 2;
})();


extern long base_compute_time_;
extern long remat_compute_time_;
extern long search_time_;
extern long cost_time_;
extern bool use_log_;
extern bool use_profile_;
extern std::unordered_map<int64_t, duration_t> compute_cost_records;
extern std::unordered_map<int64_t, size_t> memory_cost_records;
extern COMMON_API size_t memory_budget;
extern COMMON_API bool store_in_special_pool[8];
extern COMMON_API bool defrag_flag[8];
extern COMMON_API std::unordered_map<cudaStream_t, int>* stream_to_label;

#ifdef DEBUG_MODE
// extern const bool record_er_counts;        // 驱逐&重物化次数
// extern const bool record_op_recs;          // 是否记录op历史
extern bool record_er_counts;        // 驱逐&重物化次数
extern bool record_op_recs;          // 是否记录op历史

extern const bool record_cpevict_recs;     // 记录驱逐记录
extern const bool record_fragmentation;    // 记录碎片化和内存占用数据
extern const bool record_lifecycle;        // 记录ap生命周期计数分布
extern const bool record_ap_cost;          // 记录ap的cost分布
extern const bool record_dependcy;
extern const bool record_key_chain;
extern const bool trace_register_and_release;   // 追踪所有ext和ap的生命周期(适合demo debug)
extern COMMON_API const bool trace_evicted_tensor;  // 追踪驱逐算法选择的张量
extern const bool record_dcr_process;     // 记录dcr过程
extern const bool record_dcr_memory;      // 记录dcr过程中对应mem的情况

extern size_t dcr_lock_counts;

extern bool record_mem_addr;         // 是否记录内存地址
extern bool current_if_any_evicted;

extern std::atomic<size_t> evict_counts;
extern std::atomic<size_t> tensor_evict_counts;
extern std::atomic<size_t> remat_counts;
extern std::atomic<size_t> cannot_evict_counts;
extern std::atomic<size_t> destruct_counts;
extern std::atomic<size_t> tensor_destruct_counts;

void signal_handler(int sig);

#endif

// TODO: using a pool allocator might make more sense - no need to allocate and delete each pointer individually.
template<typename T>
struct EquivalentClassNode : intrusive_ptr_target {
  explicit EquivalentClassNode(const T& t) : t_unsafe(t) { }
  mutable intrusive_ptr<EquivalentClassNode> parent;
  bool is_root() {    // no parent node means root
    return !parent;
  }
  void release_resources() override {
    parent.reset();
  }
  T t_unsafe;
};

// 返回根节点的值
template<typename T>
T& get_t(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  return find_root(n)->t_unsafe;
}

template<typename T>
static void update_t(const intrusive_ptr<EquivalentClassNode<T>>& n, const T& t) {
  find_root(n)->t_unsafe = t;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> find_root(const intrusive_ptr<EquivalentClassNode<T>>& n) {
  if (n->is_root()) {
    return n;
  } else {
    n->parent = find_root(n->parent);
    return n->parent;
  }
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> merge(const std::function<T(const T&, const T&)>& merge_t,
                                            const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  if (l == r) {
    return l;
  }
  l->parent = r;
  r->t_unsafe = merge_t(l->t_unsafe, r->t_unsafe);
  return r;
}

template<typename T>
bool is_same_union(const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                   const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto l = find_root(lhs);
  auto r = find_root(rhs);
  return l == r;
}

template<typename T>
intrusive_ptr<EquivalentClassNode<T>> flat(const intrusive_ptr<EquivalentClassNode<T>>& lhs,
                                            const intrusive_ptr<EquivalentClassNode<T>>& rhs) {
  auto root = find_root(lhs);
  lhs->parent = root;
  rhs->parent = root;
  return root;
}

#pragma endregion

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;
  void release_resources() final {
    static_release_resources(value);
  }
  RefCell(const T& t) : value(t) { }  /// TODO: 这里没有std::move，传入的T是c10::intrusive_ptr<at::External, c10::detail::intrusive_target_default_null_type<at::External> >
};

template<typename T>
using Ref = intrusive_ptr<RefCell<T>>;

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) {
  ptr.reset();
}

struct CheckpointInfo {
  duration_t compute_cost;
  // Floating Point instability?
  double cost(size_t memory, size_t staleness) const {
    // TORCH_CHECK(memory > 0);
    // TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  double cost(size_t memory, size_t staleness, uintptr_t addr) const {  /// TODO: 设计一个新的计算指标
    // TORCH_CHECK(memory > 0);
    // TORCH_CHECK(staleness > 0);
    return compute_cost.count() / static_cast<double>(memory * staleness);
  }
  CheckpointInfo(duration_t compute_cost) :
    compute_cost(compute_cost) {
  }
};

// ecn represent a evicted tensor group.
// it is a set of tensor that are evicted, and if two evicted tensor are input -> output to each other,
// they must be in an ecn.
// note: we try to support removal from ecn by subtracting compute_cost and memory.
// this will create suprious connection but that should be fine empircally.
// below is an example of a suprious connection:
// a -> b, a -> c
// a, b, c got evicted so belong to a single ecn.
// a got rematerialized.
// b, c still belong to a single ecn although there is no connection.
using ecn_ptr = intrusive_ptr<EquivalentClassNode<CheckpointInfo>>;

struct Unsafe { };

#ifdef DEBUG_MODE
void printStackTrace();
#endif


}
}