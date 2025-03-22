#include <c10/cuda/CUDACachingAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/dtb/DTBManager.h>
#include <c10/core/dtb/utils.h>
#include <c10/util/CallOnce.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/static_tracepoint.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <utility>
#include <vector>

TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

#define MEM_TWIN_REC
// #define MEM_TWIN_DEBUG

// #define GMLAKE_ENABLE // GMLAKE history trace is unavailable(wrong history)
#ifdef GMLAKE_ENABLE
#include <c10/util/Backtrace.h>
#include <unordered_map>
#include <unordered_set>
#include <c10/cuda/cuda_vmm_allocator.h>
#endif

namespace c10 {

C10_DEFINE_REGISTRY(FreeCudaMemoryCallbacksRegistry, FreeMemoryCallback);

#ifdef MEM_EVENTS_REC
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h> // for getpid()

static bool last_flag = false;
static const bool log_cudaAPI = ([]() -> bool {
    const char* env = getenv("LOG_CUDAAPI");
    if(env) return (atoi(env))==1;
    else    return false;
})();
static const bool log_mem_events = ([]() -> bool {
    const char* env = getenv("LOG_MEM_EVENTS");
    if(env) return (atoi(env))==1;
    else    return false;
})();
static const std::string mem_log_prefix = ([]() -> std::string {
    const char* env = getenv("LOG_MEM_PREFIX");
    if(env) return {env};
    else    return ".";
})();

// static const bool COST_FIRST_EVICT = ([]() -> bool {
//     const char* env = getenv("COST_FIRST_EVICT");
//     if(env) return (atoi(env))==1;
//     else    return false;
// })();


struct DTRLogger {
  std::string time_prefix;
  FILE *out;

  static std::string get_time_prefix() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    pid_t pid = getpid();
    return
      std::to_string(1900+tm->tm_year) + "-" +
      std::to_string(1+tm->tm_mon) + "-" +
      std::to_string(tm->tm_mday) + "-" +
      std::to_string(tm->tm_hour) + "-" +
      std::to_string(tm->tm_min) + "-" +
      std::to_string(tm->tm_sec) + "-" +
      std::to_string(pid);
  }

  std::string get_filename(const std::string& name) {
    return mem_log_prefix + "/" + time_prefix + "-" + name + ".log";
  }

  DTRLogger() : time_prefix(get_time_prefix()) {
    std::string filename = get_filename("default");
    out = fopen(filename.c_str(), "a");  // 'a' for appending to the file
    if (!out) {
      perror("Failed to open log file");
      exit(1);
    }
  }

  ~DTRLogger() {
    if (out) {
      fclose(out);
    }
  }

  void log(const std::string& str) {
    if (out) {
      fprintf(out, "%s\n", str.c_str());
      fflush(out); // Ensure it writes immediately to the file
    }
  }

  static DTRLogger& logger() {
    static DTRLogger ret;
    return ret;
  }
};


void DTRLogMemEvents(const std::string& name, size_t size, int64_t addr) {
  std::string log_msg = "{";
  log_msg += "\"TYPE\":\"" + name + "\", ";
  log_msg += "\"SIZE\":" + std::to_string(size) + ", ";
  log_msg += "\"ADDR\":" + std::to_string(addr);
  log_msg += "}";
  DTRLogger::logger().log(log_msg);
}

void DTRLogMemSegmentStat(const std::string& name, size_t size, int64_t addr, int64_t blocks_num) {
  std::string log_msg = "{";
  log_msg += "\"TYPE\":\"" + name + "\", ";
  log_msg += "\"SIZE\":" + std::to_string(size) + ", ";
  log_msg += "\"BLOCK_NUM\":" + std::to_string(blocks_num) + ", ";
  log_msg += "\"ADDR\":" + std::to_string(addr);
  log_msg += "}";
  DTRLogger::logger().log(log_msg);
}

size_t cudaMalloc_counts = 0, cudaFree_counts = 0;
void DTRLogcudaAPIEvents(const std::string& name, size_t size, int64_t addr) {
  std::string log_msg = "{";
  log_msg += "\"TYPE\":\"" + name + "\", ";
  log_msg += "\"COUNTS\":\"" + (name=="cudaMalloc" ? std::to_string(++cudaMalloc_counts) : std::to_string(++cudaFree_counts)) + "\", ";
  log_msg += "\"SIZE\":" + std::to_string(size) + ", ";
  log_msg += "\"ADDR\":" + std::to_string(addr);
  log_msg += "}";
  DTRLogger::logger().log(log_msg);
}

void DTRLogSegmentsStats(const size_t& size, const size_t& blocks_num, const size_t& timstamp, const std::vector<void*> ptrs, const std::vector<int> status){
  std::string log_msg = "{";
  log_msg += "\"TYPE\":\"SEGMENT INFO\", ";
  log_msg += "\"SIZE\":" + std::to_string(size) + ", ";
  log_msg += "\"MEMBERS\":" + std::to_string(blocks_num) + ", ";
  log_msg += "\"ADDR\":[";
  for(size_t i=0; i<ptrs.size(); i++){
    log_msg += std::to_string(reinterpret_cast<uintptr_t>(ptrs[i]));
    if(i!=ptrs.size()-1) log_msg += + ", ";
  }
  log_msg += "], ";
  log_msg += "\"STATUS\":[";
  for(size_t i=0; i<status.size(); i++){
    log_msg += std::to_string((status[i]));
    if(i!=ptrs.size()-1) log_msg += + ", ";
  }
  log_msg += "], ";
  log_msg += "\"TIMESTAMP\":" + std::to_string(timstamp);
  log_msg += "}";
  DTRLogger::logger().log(log_msg);
}

#endif

namespace cuda {
namespace CUDACachingAllocator {
namespace Native {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks

#ifndef MORE_POOL
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
#else
static const size_t kE1Size = ([]() -> size_t {
    const char* env = getenv("E1_POOL_MAX");
    if(env) return static_cast<size_t>(atoi(env));
    else return 20971520;   // 20MiB
})();
static const size_t kE2Size = ([]() -> size_t {
    const char* env = getenv("E2_POOL_MAX");
    if(env) return static_cast<size_t>(atoi(env));
    else return 67108864;   // 64MiB
})();
// constexpr size_t kE1Size =  // used to choose pool
//     20971520; // allocations between 1 and 20 MiB may use kE1Buffer
constexpr size_t kE1Buffer =
    20971520; // "E1" allocations may be packed in 20 MiB blocks

// constexpr size_t kE2Size =  // used to choose pool 
//     268435456; // allocations between 20 and 200 MiB may use kE2Buffer 268435456 （256M)
// constexpr size_t kE2Size =  // used to choose pool 
//     67108864; // allocations between 20 and 64 MiB may use kE2Buffer

constexpr size_t kE2Buffer =
    2147483648; // "E2" allocations may be packed in 2048(8*256) MiB blocks


// constexpr size_t kMinE1Alloc =  // used to choose size 
//     20971520; // allocations between 1 and 20 MiB may use kE1Buffer
// constexpr size_t kMinE2Alloc = 
//     1024*1024*200; // allocations between 20 and 256 MiB may use kE2Buffer 

// else all use Large blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 32 and 64 MiB may use kLargeBuffer   [not use]
#endif

constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB
constexpr size_t kGranularity   =  2097152; // round up large allocations to 2 MiB (GMLake)
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

namespace {

using stream_set = ska::flat_hash_set<cuda::CUDAStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  if(stat.current<0){
    printf("error\n");
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}

#ifdef GMLAKE_ENABLE
struct EventIDCounter
{
  EventIDCounter(cudaStream_t stream):stream(stream), current_event_id(0) {}
    
  void reset() {
    std::lock_guard<std::recursive_mutex> lock(id_mutex);
    current_event_id = 0;
  }
    
  std::uint64_t next_id() {
    std::lock_guard<std::recursive_mutex> lock(id_mutex);
    
    if(current_event_id == std::numeric_limits<uint64_t>::max())
      current_event_id = 1;
    else
      current_event_id++;
        
      return current_event_id;
  }
    
  std::recursive_mutex id_mutex;
  cudaStream_t stream;
  std::uint64_t current_event_id;
};

static std::unordered_map<cudaStream_t, std::shared_ptr<EventIDCounter>> stream_id_counter;
static std::mutex counter_mutex;
struct BlockEvent {
  BlockEvent(cudaStream_t stream_in, bool record_event=false) {
    stream = stream_in;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    event_id = 0;
    released = false;
    ref_as_sync = false;
    if(record_event) record(stream);
    
  }

  void record(cudaStream_t stream_in) {

    if(stream == stream_in)
    {
      std::shared_ptr<EventIDCounter> id_counter;
      if(stream_id_counter.find(stream) == stream_id_counter.end()) {
        id_counter = std::make_shared<EventIDCounter>(stream);
        {
          std::lock_guard<std::mutex> lock(counter_mutex);
          stream_id_counter[stream] = id_counter;
        }
      } else {
        std::lock_guard<std::mutex> lock(counter_mutex);
        id_counter = stream_id_counter[stream];
      }
      
      {
        std::lock_guard<std::recursive_mutex> lock(id_counter->id_mutex);
        
        event_id = id_counter->next_id();
        C10_CUDA_CHECK(cudaEventRecord(event, stream));
      }
    }
  }

  void release_resources()
  {
    if(!ref_as_sync) {
      C10_CUDA_CHECK(cudaEventDestroy(event));
    } else {
      cudaError_t err = cudaEventQuery(event);
      if(err == cudaSuccess) {
        C10_CUDA_CHECK(cudaEventDestroy(event));
      } else if(err == cudaErrorNotReady) {
        cudaGetLastError();
        event_gc(stream, event_id, event);
      } else {
        C10_CUDA_CHECK(err);
        cudaGetLastError();
        C10_CUDA_CHECK(cudaEventDestroy(event));
      }
    }
  }

  ~BlockEvent() {
    if (!released) {
      this->release_resources();
      released = true;
    }
  }

  static uint64_t create_id()
  {
    static std::atomic<std::uint64_t> current_event_id = {0};
    static uint64_t id_max = std::numeric_limits<uint64_t>::max();

    current_event_id.compare_exchange_strong(id_max, std::uint64_t(0));
    uint64_t my_id = current_event_id.fetch_add(1);
    
    return my_id;
  }

  static void event_gc(cudaStream_t stream, uint64_t event_id, cudaEvent_t event)
  {
    using EventQueue=std::map<uint64_t, cudaEvent_t>;
    
    static std::unordered_map<cudaStream_t, EventQueue> event_pool;
    static std::mutex pool_mutex;

    {
      std::lock_guard<std::mutex> lock(pool_mutex);
      event_pool[stream][event_id] = event;
    }
    
    auto& event_queue = event_pool.at(stream);
    if(event_queue.size() > 2000)
    {
      std::lock_guard<std::mutex> lock(pool_mutex);

      for(auto it = event_queue.begin(); it != std::prev(event_queue.end());) {
        cudaEvent_t event = it->second;
        cudaError_t err = cudaEventQuery(event);
        if(err == cudaSuccess) {
          C10_CUDA_CHECK(cudaEventDestroy(event));
          it = event_queue.erase(it);
        } else {
          cudaGetLastError();
          break;
        }
      }
    }
  }
  
  cudaStream_t stream;
  cudaEvent_t event;
  uint64_t event_id;
  bool released;
  bool ref_as_sync;
};
#endif

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

#ifndef MORE_POOL

struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
};

#else

struct BlockPool {
  BlockPool(StatType bt, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        pool_type(bt),
        owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  const StatType pool_type;
  PrivatePool* owner_PrivatePool;
};

#endif

struct ExpandableSegment;

#ifdef GMLAKE_ENABLE
struct HistoryChain {
  History h;
  std::unique_ptr<HistoryChain> next; // when blocks are merged we keep records
                                      // of what used to be in the block
};
#endif

struct Block {
  int device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
#ifdef GMLAKE_ENABLE
  size_t actual_size;
#endif
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  bool mapped{true}; // is the virtual address range this Block references
                     // backed by physical pages. Always true when
                     // expandable_segment_ is null. When false
                     // This Block will be aligned to the segment size
                     // of its expandable_segment_.
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int gc_count{0}; // counter for prioritizing older / less useful blocks for
                   // garbage collection
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when cudaMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};
#ifdef GMLAKE_ENABLE
  std::unique_ptr<HistoryChain> history;
  HistoryChain* history_last{nullptr};
  std::shared_ptr<VmmSegment> vmm_segment;
  std::shared_ptr<BlockEvent> self_last_event;
#endif

  Block(
      int device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
#ifdef GMLAKE_ENABLE
        actual_size(0),
        self_last_event(std::make_shared<BlockEvent>(stream)),
#endif
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
#ifdef GMLAKE_ENABLE
        actual_size(0),
        self_last_event(std::make_shared<BlockEvent>(stream)),
#endif
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

/*
Note [Expandable Segments]

Rationale

For large (>2MB) allocations, the allocator calls cudaMalloc to get allocations
that are the same size as what the user requests. In the future, parts of these
allocations can be reused for other requests if they are free. This works well
when the program makes many requests of exactly the same size or of sizes that
even multiples of that size. Many deep learning models follow this behavior.
However, one common exception is when the batch size changes slightly from one
iteration to the next, e.g. in batched inference. When the program runs
initially with batch size N, it will make allocations appropriate for that size.
If in the future, it runs at size N - 1, the existing allocations will still be
big enough. However, if it runs at size N + 1, then it will have to make new
allocations that are slightly larger. Not all the tensors are the same size.
Some might be (N + 1)*A and others (N + 1)*A*B where A and B are some non-batch
dimensions in the model. Because the allocator reuses existing allocations when
they are big enough, some number of (N + 1)*A allocations will actually fit in
the already existing N*B*A segments, though not perfectly. As the model runs it
will partially fill up all of these segments leaving unusable free slices of
memory at the end of these segments. The allocator at some point will need to
cudaMalloc a new (N + 1)*A*B segment. If there is not enough memory, there is
now no way to recover the slices of memory that are free at the end of existing
segments. With models 50+ layers deep, this pattern might repeat 50+ times
creating many slivers.

Approach

Expandable segments allows the allocator to create a segment initially and then
expand its size later when more memory is needed. Instead of making one segment
per allocation, it tries to make one segment (per stream) that grows as
necessary. Now when the N + 1 case runs, the allocations will tile nicely into
the one large segment until it fills up. Then more memory is requested and
appended to the end of the segment. This process does not create as many slivers
of unusable memory, so it is more likely to succeed at finding this memory.

Implementation

The expandable_segments:True option is used to enable/disable this behavior. We
use cuda's low-level memory APIs, which are similar to mmap, to extend the
memory segments. These APIs separate the allocation of physical memory
(cuMemCreate) from the allocation of virtual address space (cuMemAddressReserve)
and the associate between them cuMemMap/cuMemSetAccess.

When we allocate a new segment, we allocate enough address space to map
basically the entire physical memory of the GPU (there is 256TiB of address
space), but we only map enough physical memory to handle the current amount of
memory needed by the program. As more is requested, we add more physical memory
to the segment. This can work at the granularity of GPU pages which are 2MiB
currently.

If we end up out of memory, we can unmap all the memory in our segment
corresponding to empty physical pages, and return it to CUDA for use at another
address in the segment or in a segment for a different stream.

A current limitation of CUDA's API is that physical memory
(CUmemGenericAllocationHandle) cannot be split up after it is mapped even if the
handle holds multiple GPU pages. The cost to map/unmap memory is proportional to
the number of physical memory chunks that were allocated (mapping 10 separately
allocated 2MiB pages takes 10x time compared to mapping one 20MiB physical
allocation of 10 pages).  Changing memory mappings also appears to involve at
least some synchronous actions with the GPU and so should be considered an
expensive operation. To limit overhead, we use 2MiB pages for our small pool and
20MiB pages for our large pool. Initially allocation using expandable_blocks
will be slower than cudaMalloc, though still in the milliseconds range for
mapping the entire memory.

When mapping new memory to expand the segment, we look for the lowest address at
which we can fit a new allocation by adding new pages. Normally this will be at
the end of the block. But if have previously unmapped blocks earlier in the
segment during an OOM, it will first try to fill in those gaps to keep the
segment as a single block. By allocating at the lowest address we encourage
the split up parts of the block to merge into a single block again, reducing
fragmentation potential.

Allocation of blocks in the segment uses the same best-fit heuristics of the
rest of the allocator.

Expandable blocks can be enabled/disabled throughout the run of a program. When
disabled, the allocator will not put new allocations in an expandable block.

Limitations

* Slightly slower initial memory allocation speed.
* IPC of cuda tensors (e.g. for multiprocess dataloaders) is not supported.
However, it is possible to temporarily disable (expandable_segments:False) the
bevhavior for allocator tensors that need to be used cross-process.
* CUDA runtime APIs related to sharing memory across process
(cudaDeviceEnablePeerAccess) do not work for memory allocated with cuMemMap.
Instead these mapping have to be done manually. The allocator now has an
`enablePeerAccess` method to do this.
*/

struct ExpandableSegment {
  ExpandableSegment(
      int device,
      cudaStream_t stream,
      size_t size,
      std::vector<int> peers)
      : device_(device),
        stream_(stream),
        max_handles_(0),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size),
        peers_(std::move(peers)) {
    cudaDeviceProp prop{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressReserve_(
        &ptr_, segment_size_ * max_handles_, 0ULL, 0, 0ULL));
  }
  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      CUmemGenericAllocationHandle handle = 0;
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      auto status =
          DriverAPI::get()->cuMemCreate_(&handle, segment_size_, &prop, 0);
      if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      C10_CUDA_DRIVER_CHECK(status);
      handles_.at(i) = handle;
    }
    for (auto i : c10::irange(begin, end)) {
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemMap_(
          ptr_ + i * segment_size_,
          segment_size_,
          0,
          handles_.at(i).value(),
          0ULL));
    }

    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const {
    return (char*)ptr_;
  }
  size_t size() const {
    return max_handles_ * segment_size_;
  }

  void addPeer(int device) {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

  ~ExpandableSegment() {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemAddressFree_(
        ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(int device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemSetAccess_(
        ptr_ + begin * segment_size_, (end - begin) * segment_size_, &desc, 1));
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    C10_CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (auto i : c10::irange(begin, end)) {
      CUmemGenericAllocationHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemUnmap_(
          ptr_ + segment_size_ * i, segment_size_));
      C10_CUDA_DRIVER_CHECK(DriverAPI::get()->cuMemRelease_(h));
    }
    trimHandles();
  }
  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }
  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    auto start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }
  size_t numSegments(size_t size) {
    return (size + segment_size_ - 1) / segment_size_;
  }
  size_t segmentLeft(char* p) {
    auto size = p - ptr();
    return size / segment_size_;
  }
  size_t segmentRight(char* p) {
    auto size = p - ptr();
    return numSegments(size);
  }
  SegmentRange rangeFromHandles(size_t begin, size_t end) {
    return SegmentRange(
        ptr() + segment_size_ * begin, segment_size_ * (end - begin));
  }
  int device_;
  cudaStream_t stream_;
  CUdeviceptr ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<CUmemGenericAllocationHandle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<int> peers_;
};
#else
struct ExpandableSegment {
  ExpandableSegment(
      int device,
      cudaStream_t stream,
      size_t size,
      const std::vector<int>& peers) {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  char* ptr() const {
    return nullptr;
  }
  size_t size() const {
    return 0;
  }
  void addPeer(int device) {}
};
#endif

#ifdef MEM_TWIN_REC
using c10::dtb::time_t;

// 自定义比较器，根据 stream以及地址顺序 进行排序
struct CompareBlockInSegment {
  bool operator()(const Block* a, const Block* b) const {
      if (a->stream != b->stream) {
        return (uintptr_t)a->stream < (uintptr_t)b->stream;
      }
      return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
    }
};

struct SegmentTwin {
  // ska::flat_hash_set<Block*> blocks;
  std::set<Block*, CompareBlockInSegment> blocks;
  bool is_small = false;
  bool evictable = true;        // change when flush_cost
  bool all_can_mv = true;       // change when flush_cost
  size_t total_size;
  size_t allocated_size = 0;
  time_t last_change_time;
  double max_evict_cost=0;      // represent cost of evicting this segment
  size_t can_free_size=0;
  float frag_ratio=0;

  SegmentTwin* pre{nullptr};
  SegmentTwin* next{nullptr};

  SegmentTwin(Block* head);
  void insert(Block* block);
  void erase(Block* block);
  bool empty();
  inline void flush_frag_ratio();
  inline Block* flush_cost(time_t cur, size_t need_size, cudaStream_t stream);
};

SegmentTwin::SegmentTwin(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
#ifndef MORE_POOL
  is_small = head->pool->is_small;
#else
  is_small = head->pool->pool_type==StatType::SMALL_POOL;
#endif
  last_change_time = std::chrono::system_clock::now();
  total_size = head->size;
  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    // blocks.emplace_back(curr);
    blocks.insert(curr);
  }
}

void SegmentTwin::insert(Block* block) {
  // blocks.emplace_back(block);
  blocks.insert(block);
  // allocated_size += block->allocated ? block->size : 0;
  evictable = true;
  // last_change_time = std::chrono::system_clock::now();   // updated in Manager
}

void SegmentTwin::erase(Block* block) {
  // int index=0;
  // bool find_flag = false;
  // for(; index < blocks.size(); index++){
  //   if(block==blocks[index]){
  //     find_flag = true;
  //     break;
  //   }
  // }
  // if(find_flag) blocks.erase(blocks.begin() + index);
  blocks.erase(block);
  // allocated_size -= block->allocated ? block->size : 0;
  evictable = true;
}

inline void SegmentTwin::flush_frag_ratio() {
  allocated_size = 0;
  for(auto& bit: blocks) {
    allocated_size += bit->allocated ? bit->size : 0;
  }
  frag_ratio = 1 - (static_cast<float>(allocated_size) / static_cast<float>(total_size));
}

/**
 * find out whether this segment can be evicted to satisfy need_size
*/
inline Block* SegmentTwin::flush_cost(time_t cur, size_t need_size, cudaStream_t stream) {
    auto *pm = c10::dtb::getDTBPoolManager();
    can_free_size = 0;
    max_evict_cost = 0;
    size_t unallocated_size = 0;
    bool can_evict = false;
    // all_can_mv = true;
    Block* evict_start = nullptr;
    for(const auto& bit: blocks){
      // 似乎没有必要，其他流的张量释放后该空间也能被复用
      auto ptr = bit->ptr;
      auto res = pm->get_ap_by_ptr(ptr);
      if(res.second){
        if(auto sap = res.first.lock()){
          if(sap->evictable()&&!sap->is_evicted){
            if(!evict_start) evict_start = bit;
            max_evict_cost += sap->cost(cur);
            can_free_size += sap->memory;
            can_evict = true;
          }else{
            // all_can_mv = false;
            if(need_size != 0) {  // cannot evict, reset stats
              evict_start = nullptr;
              max_evict_cost = 0;
              can_free_size = 0;
            }
            if(total_size==need_size){    // avoid fragmentation
              can_evict = false;
              break;
            }
          }
        }
      } else {
        // all_can_mv = false;
        if(need_size != 0) {
          if(bit->allocated) {  // no record memory block, reset stats
            evict_start = nullptr;
            max_evict_cost = 0;
            can_free_size = 0;
          } else {  // unallocated memory block
            // evict_start = nullptr;
            // max_evict_cost = 0;
            can_free_size += bit->size;
            unallocated_size += bit->size;
          }
        }
      }
      if(need_size != 0 && can_free_size >= need_size) break;
    }
    frag_ratio = static_cast<float>(unallocated_size) / static_cast<float>(total_size);
    if(can_free_size < need_size || frag_ratio == 1.f) 
      can_evict = false;
    evictable = can_evict;
    return evict_start;
}

bool SegmentTwin::empty(){
  return blocks.empty();
}

// 自定义比较器，根据 last_change_time 进行排序
struct CompareSegment {
  bool operator()(const SegmentTwin* lhs, const SegmentTwin* rhs) const {
      if(lhs->last_change_time != rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
      return lhs->blocks.size() < rhs->blocks.size();
  }
};

// 自定义比较器，根据 cost 进行排序
struct CompareCostSegment {
  bool operator()(const SegmentTwin* lhs, const SegmentTwin* rhs) const {
      if(lhs->max_evict_cost != rhs->max_evict_cost) return lhs->max_evict_cost < rhs->max_evict_cost;
      if(lhs->last_change_time != rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
      return lhs->blocks.size() < rhs->blocks.size();
  }
};

class SegmentManager {
private:
  std::map<size_t, std::vector<SegmentTwin*>> size_map;
  ska::flat_hash_map<void*, SegmentTwin*> blocks2segment;
  std::vector<Block*> free_block_buffer;
  SegmentTwin* defrag_seg = nullptr;

public:

  SegmentTwin* get_defrag_seg() { return defrag_seg; };
  void set_defrag_seg(SegmentTwin* seg) { defrag_seg = seg; };

  void add_block2segment(Block* block, SegmentTwin* seg) {
    blocks2segment[block->ptr] = seg;
    seg->insert(block);   // update last_change_time
    seg->last_change_time = std::chrono::system_clock::now();
    // update(seg);       // vector存储size_map并不需要更新排序
  }

  SegmentTwin* get_segment_of_block(void* ptr, bool remove = false) {
    auto it = blocks2segment.find(ptr);
    if (it == blocks2segment.end()) {
      return nullptr;
    }
    SegmentTwin* seg = it->second;
    if (remove) {
      blocks2segment.erase(it);
    }
    return seg;
  }

  /* Insert a new created SegmentTwin */
  void insert(SegmentTwin* new_segment) {
    // time_t begin = std::chrono::system_clock::now();
    auto it = size_map.find(new_segment->total_size);
    if(it==size_map.end()){
      // std::set<SegmentTwin*, CompareSegment> segments;
      std::vector<SegmentTwin*> segments;
      segments.emplace_back(new_segment);
      size_map[new_segment->total_size] = segments;
    }else{
      size_map[new_segment->total_size].emplace_back(new_segment);
    }
    // time_t end = std::chrono::system_clock::now();
    // auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "[insert]" << time_cost << "us" << std::endl;
  }

  void erase(SegmentTwin* segment) {
    // time_t begin = std::chrono::system_clock::now();
    auto it = size_map.find(segment->total_size);
    TORCH_INTERNAL_ASSERT(it!=size_map.end());
    // it->second.erase(segment);
    for(size_t i=0; i<it->second.size(); i++){
      if(it->second[i]==segment){
        it->second[i] = it->second.back();
        it->second.pop_back();
        break;
      }
    }
    if(it->second.empty()){
      size_map.erase(it);
    }
    // time_t end = std::chrono::system_clock::now();
    // auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    // std::cout << "[erase]" << time_cost << "us" << std::endl;
  }

  void rearrange_on_site(SegmentTwin* segment) {
    /// 整个函数都应该处于一个锁的保护下，所有对该segment的操作应该都先检查是否在锁定状态
    /// TODO: 就地整理，出于安全角度考虑 应该继续走torch本身的接口，保证计算语义不变
    /// ref: get_free_block

    /// 1. release / offload current block -- directly call cptc func
    ///   a. version 1, naive release stragety
    ///   b. version 2, find the best adjust stragety
    std::vector<c10::intrusive_ptr<c10::dtb::AliasPool>> to_release;
    auto *pm = c10::dtb::getDTBPoolManager();
    bool if_add = false;
    for(auto &bit: segment->blocks) {
      if(!bit->allocated) {
        if_add = true;
      }else{
        auto res = pm->get_ap_by_ptr(bit->ptr);
        if(res.second) {
          if(auto sap = res.first.lock()) {
            if(sap->evictable())
              to_release.push_back(sap);
          }
        }
      }
    }
    // if all sap can be evicted, defragmentation will work?
    std::vector<c10::dtb::strong> to_recovery;
    for(auto& sap: to_release) {
      for(auto& wcell: sap->tensors) {
        if(auto scell = wcell.lock()) {
          if(scell->defined) {
            to_recovery.push_back(scell);
          }
        }
      }
      sap->evict(0);
    }
    
    /// 2. get free block -- use current blocks, use a buffer to intercept malloc()
    set_defrag_seg(segment);
    c10::dtb::defrag_flag[c10::cuda::current_device()] = true;


    /// 3. re alloc block -- 
    ///   a. maybe cudaMemcpyAsync + change tensor data_ptr which in storage
    ///   b. like proactive process, use another stream in torch or in c++  https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    /// c10/core/Storage.h:102
    for(auto& scell: to_recovery) { // TODO: clone or recompute
      scell->get();
    }

    c10::dtb::defrag_flag[c10::cuda::current_device()] = false;
    set_defrag_seg(nullptr);

  }

  void rearrange_to_segment() {
    /// 1. release / offload current block

    /// 2. get target block

    /// 3. re alloc block

  }

  void display_segments(){
#ifdef MEM_FIRST_EVICT
    auto *pm = c10::dtb::getDTBPoolManager();
    // printf("[CHECK p2ap] %ld\n",pm->get_p2ap_size());
    size_t cans = 0, cannot = 0, no_rec = 0, frees = 0, norecButInAps = 0;
    size_t cans_size = 0, cannot_size = 0, no_rec_size = 0, frees_size = 0;
    int device_id = 0;
    for(const auto& it: size_map){
      size_t segment_size = it.first;
      for(const auto& seg_it: it.second){
        std::vector<void*> data_ptrs;
        std::vector<int> if_evictable;
        TORCH_INTERNAL_ASSERT(seg_it);
        if(seg_it->blocks.empty()) { 
          printf("[EMPTY SEG]\n"); 
          frees_size += seg_it->total_size; 
          continue; 
        } // frees_size += seg_it->total_size;
        for(auto pit = seg_it->blocks.begin(); pit!=seg_it->blocks.end(); pit++){
          TORCH_INTERNAL_ASSERT((*pit)->ptr);
          auto *ptr = (*pit)->ptr;
          data_ptrs.emplace_back(ptr);
          device_id = (*pit)->device;
          if((*pit)->allocated){

            auto res = pm->get_ap_by_ptr(ptr);
            if(res.second){ // ptr in p2ap
              auto sap = res.first.lock();
              if(sap.defined()){
                TORCH_INTERNAL_ASSERT(reinterpret_cast<void*>(sap->addr)==(*pit)->ptr);
                if(sap->is_evicted){    // ap is evicted
                  TORCH_INTERNAL_ASSERT(false, "ap status is not consistent with block.")
                }else{
                  if_evictable.emplace_back(sap->evictable() ? 1 : 0);  // 1-can evict 0-cannot
                  if(sap->evictable()) { cans++; cans_size+=sap->memory; } 
                  else { cannot++; cannot_size += sap->memory; }
                }
              }
              else{  // ap is invalid, block is allocated but ap is released
                if_evictable.emplace_back(-2);
                cannot++;
                cannot_size += (*pit)->size;
              }
            }else{  // block is allocated but not in p2ap
              if_evictable.emplace_back(-2);  // no record ptr but used
              no_rec++;
              no_rec_size += (*pit)->size;
              bool if_inaps = pm->check_ptr_in_aps((*pit)->device, reinterpret_cast<uintptr_t>((*pit)->ptr));
              if(if_inaps) norecButInAps++;
            }

          } 
          else{ // block is free
            if_evictable.emplace_back(-1);  // free block
            frees++;
            frees_size += (*pit)->size;
          }

        }
        size_t blocks_num = seg_it->blocks.size();
        auto last_time = seg_it->last_change_time;
        auto duration = last_time.time_since_epoch();
        size_t millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();  // timestamp
        // DTRLogSegmentsStats(segment_size, blocks_num, millis, ptr);
        DTRLogSegmentsStats(segment_size, blocks_num, millis, data_ptrs, if_evictable);
      }
    }
    printf("[CHECK SEGMENTS STATIC] total_p2ap:%ld(device-%d:%ld) cans:%ld(%ld) cannot:%ld(%ld) no_rec:%ld(%ld)(%ld) frees:%ld(%ld)\n", 
      pm->get_p2ap_size(), device_id, pm->get_aps_size(device_id),
      cans, cans_size, cannot, cannot_size,
      no_rec, no_rec_size, norecButInAps, frees, frees_size);
#endif
  }

  bool auto_evict(size_t need_size, int device, cudaStream_t stream, bool if_gap=false) {
    if((c10::dtb::current_memory(device)+need_size) < c10::dtb::memory_budget) return true;

    long search_time_ = 0, search_size = 0;
    size_t before_allocated = c10::dtb::current_memory(device);
    time_t pre = std::chrono::system_clock::now();
    // size_t gap_size = c10::dtb::memory_budget > before_allocated ? (c10::dtb::memory_budget - before_allocated) : (before_allocated - c10::dtb::memory_budget);
    size_t gap_size = c10::dtb::memory_budget > before_allocated ? need_size : (before_allocated - c10::dtb::memory_budget + need_size);
    size_t low_cost_size = 0;
    bool gap_flag = if_gap;
    Block* evict_strat = nullptr;
    
    auto *pm = c10::dtb::getDTBPoolManager();
    size_t released_size = 0;

    std::vector<SegmentTwin*> to_evict_segments;
    std::vector<SegmentTwin*> to_defrag_segments;
    
    constexpr const size_t check_scale = 5;
    constexpr const double stop_threshold = 1e-10;

    /**
     * Traverse each segment to find the lowest cost is unnecessay and heavy overhead.
     * In fact, a suitable threshold of cost for early stop is beneficial to recude searching time
     * and it's enough for efficient eviction.
     * However, need_size bigger than the biggest size cannot be satisfied by evicting smaller segments.
     * Then a bigger segment will cause the over budget(gap lag, gap_size is much larger than need_size)
     * In this case, reverse traversal is used to find some big and cheap tensor to evict and avoid from small but expensive tensor.
     * 
    */
    double cur_min_cost = 1e5;
    if(gap_flag) {  // avoid from evict tiny tensors && for over budget scenario
      int size_flag = 0;
      for(auto size_it=size_map.rbegin(); size_it!=size_map.rend(); ++size_it){
        for(auto &seg: size_it->second){
          search_size++;
          seg->flush_cost(pre, 0, stream);    // reverse order, so need size is zero 
          // seg->all_can_mv = all_can_move(seg); /// TODO: 这里是否有必要进行整理？
          if(seg->evictable) {
            to_evict_segments.emplace_back(seg);
            cur_min_cost = std::min(cur_min_cost, seg->max_evict_cost);
            if(seg->max_evict_cost < stop_threshold) low_cost_size += seg->can_free_size;
          }
          float mv_trigger_threshold = size_flag > 1 ? 0.8 : 0.4;
          // if(seg->all_can_mv && seg->frag_ratio >= mv_trigger_threshold && seg->frag_ratio < 1) {
          //   to_defrag_segments.emplace_back(seg);
          // }
          if(low_cost_size > gap_size) 
            break;
        }
        size_flag++;
      }
    } else {
      for(auto size_it=size_map.lower_bound(need_size); size_it!=size_map.end(); size_it++){
        for(auto &seg: size_it->second){
          search_size++;
          auto *es = seg->flush_cost(pre, need_size, stream);
          if(seg->evictable) {
            to_evict_segments.emplace_back(seg);
            cur_min_cost = std::min(cur_min_cost, seg->max_evict_cost);
            if(seg->max_evict_cost < stop_threshold) {
              evict_strat = es;
              low_cost_size = seg->can_free_size;
            }
          }
          if(low_cost_size > need_size) 
            break;
        }
      }
    }

    // printf("search size_map time:%ld, segs:%ld\n", (std::chrono::system_clock::now() - pre).count(), search_size);
    // need_size just clear from evict_start
    if(!gap_flag && evict_strat){
      Block *to_evict = evict_strat;
      while(to_evict) {
        if(to_evict->allocated){
          auto *ptr = to_evict->ptr;
          auto res = pm->get_ap_by_ptr(ptr);
          if(res.second) {
            if(auto sap = res.first.lock()) {
              if(sap->evictable()){
                sap->evict(0);
                released_size += sap->memory;
                if((c10::dtb::current_memory(device)+need_size) < c10::dtb::memory_budget)  // need_size is satisfied
                #ifdef DEBUG_MODE
                  if(c10::dtb::trace_evicted_tensor){
                    time_t post = std::chrono::system_clock::now();
                    search_time_ += std::chrono::duration_cast<std::chrono::microseconds>(post - pre).count();
                    printf("Use Ptr to Evict, need size:%ld single search time:%ld us, segs:%ld, before alloc:%ld, release:%ld, after release: %ld\n", need_size, search_time_, search_size,
                      before_allocated, released_size, c10::dtb::current_memory(device));
                  }
                  if(c10::dtb::record_dependcy) {
                    sap->update_dependency();
                  }
                #endif
                  return true;
              }
            } 
          }
        }else{
          released_size += to_evict->size;
        }
        to_evict = to_evict->next;
      }
    }else {
      // printf("[No suitable segment] gap_flag:%d, gap_size:%ld, low_cost_size:%ld, need_size:%ld\n", gap_flag, gap_size, low_cost_size, need_size);
    }


    // if gap_flag or clear from evict_start failed, sort segments and then exectute eviction

    std::sort(to_evict_segments.begin(), to_evict_segments.end(), [](SegmentTwin* lhs, SegmentTwin* rhs) {
        if(lhs->max_evict_cost != rhs->max_evict_cost) return lhs->max_evict_cost < rhs->max_evict_cost;
        if(lhs->last_change_time != rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
        return lhs->blocks.size() < rhs->blocks.size();
      });

    for(auto& to_evict: to_evict_segments){
      if((c10::dtb::current_memory(device)+need_size) < c10::dtb::memory_budget){
        break;
      }

      for(auto bit = to_evict->blocks.begin(); (bit != to_evict->blocks.end()); bit++){
        if((*bit)->allocated){
          auto *ptr = (*bit)->ptr;
          auto res = pm->get_ap_by_ptr(ptr);
          if(res.second) {
            if(auto sap = res.first.lock()) {
              if(sap->evictable()){
              #ifdef DEBUG_MODE
                if(c10::dtb::trace_evicted_tensor){
                  printf("[expected total:%lf]evict cost:%lf mem:%ld|", to_evict->max_evict_cost*1e9, sap->cost(pre)*1e9, sap->memory);
                  released_size += sap->memory;
                }
              #endif
                sap->evict(0);
                if((c10::dtb::current_memory(device)+need_size) < c10::dtb::memory_budget) break;
              }
              // }
              // else {
              //   if(sap->head_remat)
              //     printf("choosed evict-cost:%lf mem:%ld but %d %d %ld %ld %ld|", sap->cost(pre)*1e9, sap->memory, sap->evictable()?1:0, sap->is_evicted?1:0,
              //       sap->lock_count, sap->external_count, sap->remat_count);
              // }
            } 
          }
        }
      }
      
    }

    // for(auto& seg: to_defrag_segments) {
    //   move_segment_to_others(seg);
    // }

    bool if_satisfy = (c10::dtb::current_memory(device)+need_size) < c10::dtb::memory_budget;

  #ifdef DEBUG_MODE
    if(c10::dtb::trace_evicted_tensor){
      if(if_satisfy){
        time_t post = std::chrono::system_clock::now();
        search_time_ += std::chrono::duration_cast<std::chrono::microseconds>(post - pre).count();
        printf("Use Set to Evict[gap:%d] need size:%ld single search time:%ld us, segs:%ld, before alloc:%ld, release:%ld, after release: %ld\n", gap_flag?1:0, 
          need_size, search_time_, search_size, before_allocated, released_size, c10::dtb::current_memory(device));
      }
      else {
        time_t post = std::chrono::system_clock::now();
        search_time_ += std::chrono::duration_cast<std::chrono::microseconds>(post - pre).count();
        printf("[out of budget] Use Set to Evict[gap:%d], need size:%ld single search time:%ld us, segs:%ld, before alloc:%ld, release:%ld, after release: %ld\n", gap_flag?1:0, 
          need_size, search_time_, search_size, before_allocated, released_size, c10::dtb::current_memory(device));
      }
    }
  #endif

    return if_satisfy;

  }

  bool all_can_move(SegmentTwin* seg) {
    auto *pm = c10::dtb::getDTBPoolManager();
    bool all_in_rec = true;
    for(auto& bit: seg->blocks) {
      if(bit->allocated){
        auto *ptr = bit->ptr;
        auto res = pm->get_ap_by_ptr(ptr);
        if(!res.second) {
          all_in_rec = false;
        }else {
          if(auto sap = res.first.lock()) {
            // if(sap->lock_count > 0) {
            //   all_in_rec = false;
            // }
          }else{
            all_in_rec = false;
          }
        }
        if(!all_in_rec) break;
      }
    }
    return all_in_rec;
  }

  size_t move_segment_to_others(SegmentTwin* target_seg){
    auto *pm = c10::dtb::getDTBPoolManager();
    c10::dtb::move_defrag_flag[current_device()] = true;
    c10::dtb::move_defrag_seg_ptr[current_device()] = (void*)target_seg;
    size_t moved_size = 0;
    for(auto& bit: target_seg->blocks) {
      if(bit->allocated){
        auto *ptr = bit->ptr;
        auto res = pm->get_ap_by_ptr(ptr);
        if(res.second) {
          if(auto sap = res.first.lock()) {
            sap->clone_and_reset();
            moved_size += sap->memory;
          } 
        }
        /// 应当在调用该函数前保证target_seg不存在这种情况
        // else {
        //   std::cout << "[NOT IN]move_for_defrag meets tensor in block but not in p2ap: " << bit->ptr << ", " << bit->size << "\n";
        // }
      }
    }
    c10::dtb::move_defrag_flag[current_device()] = false;
    c10::dtb::move_defrag_seg_ptr[current_device()] = nullptr;
    return moved_size;
  }

  
  void display_all_not_in_ap(int device) {
    for(auto it = size_map.rbegin(); it != size_map.rend(); it++) {
      auto *pm = c10::dtb::getDTBPoolManager();
      for(auto& target_seg: it->second) {

        for(auto& bit: target_seg->blocks) {
          if(bit->allocated){
            auto *ptr = bit->ptr;
            auto res = pm->get_ap_by_ptr(ptr);
            if(res.second) {
              if(auto sap = res.first.lock()) {
                // sap->clone_and_reset(need_size);
                // pm->update_ap(sap, sap->addr);
                // allocted_size_in += sap->memory;
              } 
            }else {
              std::cout << "[NOT IN]display_all_not_in_ap meets tensor in block but not in p2ap: " << bit->ptr << ", " << bit->size << "\n";
            }
          }
        }

      }
    }
  }

  
  bool move_for_defrag(size_t need_size, int device) {
    // time_t begin = std::chrono::system_clock::now();

    std::vector<SegmentTwin*> candidates_segments;
    auto it = size_map.lower_bound(need_size);
    if(it==size_map.end()) return false;

    auto *pm = c10::dtb::getDTBPoolManager();

    candidates_segments = it->second;

    for(auto& seg: candidates_segments) {
      seg->flush_frag_ratio();
    }

    // std::sort(candidates_segments.begin(), candidates_segments.end(), [](SegmentTwin* lhs, SegmentTwin* rhs) {
    //   if(lhs->frag_ratio != rhs->frag_ratio) return lhs->frag_ratio > rhs->frag_ratio;
    //   if(lhs->last_change_time != rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
    //   return lhs->blocks.size() < rhs->blocks.size();
    // });
    std::sort(candidates_segments.begin(), candidates_segments.end(), [](SegmentTwin* lhs, SegmentTwin* rhs) {
      // 先处理frag_ratio不等于1的情况，按照原来的降序逻辑比较
      if (lhs->frag_ratio < 1 && rhs->frag_ratio < 1) {
          if (lhs->frag_ratio!= rhs->frag_ratio) return lhs->frag_ratio > rhs->frag_ratio;
          if (lhs->last_change_time!= rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
          return lhs->blocks.size() < rhs->blocks.size();
      }
      // 如果lhs的frag_ratio等于1，而rhs的小于1，把rhs排在前面（也就是让等于1的往后排）
      // if (lhs->frag_ratio == 1 && rhs->frag_ratio < 1) return false;
      // // 如果rhs的frag_ratio等于1，而lhs的小于1，把lhs排在前面
      // if (lhs->frag_ratio < 1 && rhs->frag_ratio == 1) return true;
      // 剩下就是两者的frag_ratio都等于1的情况，按照原来后续的逻辑继续比较（这里其实可以保持和之前一样的比较逻辑，不过为了完整清晰展示思路写出来）
      if (lhs->last_change_time!= rhs->last_change_time) return lhs->last_change_time < rhs->last_change_time;
      return lhs->blocks.size() < rhs->blocks.size();
    });

    if(c10::dtb::record_move_defrag)
      std::cout << "[CHECK SORTED size:" << need_size << "] sorted frag_ratio:";
    SegmentTwin* target_seg = nullptr;
    for(auto& seg: candidates_segments) {
      if(c10::dtb::record_move_defrag)
        std::cout << "(" << seg->frag_ratio << ", " << seg->blocks.size() 
          << ", " << seg->allocated_size << ", " << (*seg->blocks.begin())->ptr << ") ";

      if(seg->frag_ratio == 1) {
        target_seg = nullptr;
        break;
      }

      if(all_can_move(seg) && seg->frag_ratio != 0) {
        target_seg = seg;
        break;
      }

    }
#ifdef DEBUG_MODE
    if(c10::dtb::record_move_defrag)
      std::cout << "\n";
    size_t before_mv = c10::dtb::current_memory(device);
    size_t allocted_size_in = 0;
#endif
    // if(target_seg->frag_ratio < 0.6 || target_seg->frag_ratio == 1.f) return false;
    if(!target_seg) return false;    /// TODO: if cancel the move when the frag ratio is at a low level 
    
    if(c10::dtb::record_move_defrag) {
      std::string seg_stat = "";
      for(auto& bit: target_seg->blocks) {  // 把是否分配的情况转换为字符串
        if(bit->allocated) seg_stat += "1";
        else seg_stat += "0";
      }
      std::cout << "move_for_defrag choosed frag: total-" << target_seg->total_size 
        << ", alloc-" << target_seg->allocated_size << ", frag_ratio: " << target_seg->frag_ratio 
        << ", stat: " << seg_stat << ", addr: " << (*target_seg->blocks.begin())->ptr << std::endl;
    }
    allocted_size_in = move_segment_to_others(target_seg);
#ifdef DEBUG_MODE
    if(c10::dtb::record_move_defrag){
      std::string seg_stat = "";
      for(auto& bit: target_seg->blocks) {  // 把是否分配的情况转换为字符串
        if(bit->allocated) seg_stat += "1";
        else seg_stat += "0";
      }
      target_seg->flush_frag_ratio();
      std::cout << "[after clone_and_reset] seg ptr: " << (*target_seg->blocks.begin())->ptr <<  ", size: " << target_seg->total_size
                << ", frag ratio: " 
                << target_seg->frag_ratio << "(" << target_seg->allocated_size
                << "), before mv: "
                << before_mv/1024/1024 << "MB, after mv: " << c10::dtb::current_memory(device)/1024/1024 << "MB, record inner size: "
                << allocted_size_in/1024/1024 << "MB, stat: " << seg_stat << std::endl;
    }
#endif

  // time_t end = std::chrono::system_clock::now();
  // auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  // std::cout << "[sMigrate]" << time_cost << "us" << std::endl;


  return true;

  }

};


#endif

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  int device = 0;
  cudaStream_t stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int gc_count = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cudaMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

static bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}
static bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

#ifdef GMLAKE_ENABLE
static bool BlockComparator(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct BlockEventOrderComparator {
  using BlockPtr=Block*;

  bool operator()(const BlockPtr a, const BlockPtr b) const {
    if(!a->self_last_event && !b->self_last_event) {
      if(a->size != b->size) {
        return a->size < b->size;
      }
  
      return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
    }
    
    if(!a->self_last_event) {
      return true;
    }
  
    if(!b->self_last_event) {
      return false;
    }


    if(a->self_last_event->event_id != b->self_last_event->event_id) {
        return a->self_last_event->event_id < b->self_last_event->event_id;
    }

    
    if(a->size != b->size) {
      return a->size < b->size;
    }
  
    return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
  }
};

using EventOrderedBlockSet=std::set<Block*, BlockEventOrderComparator>;
using SetIterator=EventOrderedBlockSet::iterator;

struct BlockEventOrderPool
{
  BlockEventOrderPool():pool_size(0) {}
    
  void insert(Block* block) {
    if(blocks.count(block) == 0) {
      blocks.insert(block);
      pool_size += block->size;
    }
  }
    
  bool erase(Block* block) {
    if(blocks.count(block)) {
      blocks.erase(block);
      pool_size -= block->size;
            
      return true;
    } else {
      GMLAKE_INFO(" warning block %p, block ptr %p of size %lu not found in pool", block, block->ptr, block->size);
      return false;
    }
  }

  SetIterator erase(SetIterator it) {
    if(blocks.count(*it)) {
      pool_size -= (*it)->size;
            
      return blocks.erase(it);
    } else {
      GMLAKE_INFO(" warning block %p, block ptr %p of size %lu not found in pool", (*it), (*it)->ptr, (*it)->size);
      return blocks.end();
    }
  }
    
    
  EventOrderedBlockSet blocks;
  size_t pool_size;
};

#endif

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(cudaSuccess) {}

  int device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  cudaError_t err;
};

#ifdef GMLAKE_ENABLE
// deprecated, this is used in torch2.0
int trimHistoryBefore(Block* block, void* point) {
  int n = 0;
  while (block->history && block->history->h.addr < point) {
    block->history = std::move(block->history->next);
    ++n;
  }
  if (!block->history) {
    block->history_last = nullptr;
  }
  return n;
}
#endif

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA graphs helper
struct PrivatePool {
  PrivatePool()
      : use_count(1),
        cudaMalloc_count(0),
#ifndef MORE_POOL
      large_blocks(/*small=*/false, this),
      small_blocks(/*small=*/true, this) {}
#else
      large_blocks(StatType::LARGE_POOL, this),
      small_blocks(StatType::SMALL_POOL, this) {}
#endif

  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // Number of live graphs using this pool
  int use_count;
  // Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critial though,
  // I'd rather not add more logic to it.
  BlockPool large_blocks;
  BlockPool small_blocks;
};

BlockState::BlockState(Block* block)
    : stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count(block->gc_count) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
};

SegmentState::SegmentState(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
#ifndef MORE_POOL
  is_small = head->pool->is_small;
#else
  is_small = head->pool->pool_type==StatType::SMALL_POOL;
#endif

  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(std::move(pool_id)) {
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
#endif
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  } else {
    // It's ok to capture cudaMallocs, as long as we never cudaFree those
    // addresses before replay.
    // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  }
#endif
}

} // anonymous namespace
} // namespace Native

// Environment config parser
// Defined here, rather than its own .cpp file,
// because parseArgs needs to know kLargeBuffer.
// Defined outside namespace Native because it's not Native-specific.
class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
#ifndef PYTORCH_C10_DRIVER_API_SUPPORTED
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size) {
    size_t log_size = (63 - llvm::countLeadingZeros(size));

    // Our intervals start at 1MB and end at 64GB
    const size_t interval_start =
        63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
    const size_t interval_end =
        63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
    TORCH_CHECK(
        (interval_end - interval_start == Native::kRoundUpPowerOfTwoIntervals),
        "kRoundUpPowerOfTwoIntervals mismatch");

    int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

    index = std::max(0, index);
    index = std::min(
        index, static_cast<int>(Native::kRoundUpPowerOfTwoIntervals) - 1);
    return instance().m_roundup_power2_divisions[index];
  }

  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_expandable_segments(false) {
    m_roundup_power2_divisions.assign(Native::kRoundUpPowerOfTwoIntervals, 0);
  }

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<bool> m_expandable_segments;
};

void CachingAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i].compare(std::string(1, c)) == 0,
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t CachingAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > Native::kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        Native::kLargeBuffer / (1024 * 1024),
        "");
    val1 = std::max(val1, Native::kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (config[i].compare("[") == 0) {
      size_t last_index = 0;
      while (++i < config.size() && config[i].compare("]") != 0) {
        const std::string& val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 ",
            "");

        if (val1.compare(">") == 0) {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (config[i + 1].compare("]") != 0) {
          consumeToken(config, ++i, ',');
        }
      }
    } else { // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseAllocatorConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_cudaMallocAsync) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == "cudaMallocAsync")),
        "Unknown allocator backend, "
        "options are native and cudaMallocAsync");
    used_cudaMallocAsync = (config[i] == "cudaMallocAsync");
    if (used_cudaMallocAsync) {
#if CUDA_VERSION >= 11040
      int version = 0;
      C10_CUDA_CHECK(cudaDriverGetVersion(&version));
      TORCH_CHECK(
          version >= 11040,
          "backend:cudaMallocAsync requires CUDA runtime "
          "11.4 or newer, but cudaDriverGetVersion returned ",
          version);
#else
      TORCH_CHECK(
          false,
          "backend:cudaMallocAsync requires PyTorch to be built with "
          "CUDA 11.4 or newer, but CUDA_VERSION is ",
          CUDA_VERSION);
#endif
    }
    TORCH_INTERNAL_ASSERT(
        config[i] == get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  } else {
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(Native::kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("roundup_power2_divisions") == 0) {
      i = parseRoundUpPower2Divisions(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("backend") == 0) {
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
    } else if (config[i] == "expandable_segments") {
      used_native_specific_option = true;
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() && (config[i] == "True" || config[i] == "False"),
          "Expected a single True/False argument for expandable_segments");
      m_expandable_segments = (config[i] == "True");
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

static std::string reportProcessMemoryInfo(int device) {
#ifdef PYTORCH_C10_DRIVER_API_SUPPORTED
  static c10::once_flag nvml_init;
  c10::call_once(nvml_init, [] {
    TORCH_INTERNAL_ASSERT(NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_());
  });

  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  char pci_id[80];
  snprintf(
      pci_id,
      sizeof(pci_id),
      NVML_DEVICE_PCI_BUS_ID_FMT,
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);

  nvmlDevice_t nvml_device = nullptr;
  TORCH_INTERNAL_ASSERT(
      NVML_SUCCESS ==
      DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_(
          pci_id, &nvml_device));

  std::vector<nvmlProcessInfo_v1_t> procs(8);
  unsigned int size = procs.size();
  nvmlReturn_t r;
  while ((r = DriverAPI::get()->nvmlDeviceGetComputeRunningProcesses_(
              nvml_device, &size, procs.data())) ==
         NVML_ERROR_INSUFFICIENT_SIZE) {
    procs.resize(size);
  }
  unsigned int self_pid = getpid();
  std::stringstream ss;
  TORCH_INTERNAL_ASSERT(NVML_SUCCESS == r);
  ss << "";
  for (auto i : c10::irange(size)) {
    auto& proc = procs[i];
    if (self_pid == proc.pid) {
      ss << "Including non-PyTorch memory, this process";
    } else {
      ss << "Process " << proc.pid;
    }
    ss << " has " << format_size(proc.usedGpuMemory) << " memory in use. ";
  }
  return ss.str();
#else
  return "";
#endif
}

namespace Native {

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;


#ifdef GMLAKE_ENABLE
// unallocated cached blocks larger than 64 MB
  //BlockPool huge_blocks;

  // fused blocks that has been mapped to fragment blocks in size order
  BlockPool free_fused_blocks;
  
  // fused blocks that has been mapped to fragment blocks in release order
  std::unordered_map<cudaStream_t, BlockEventOrderPool> free_fused_blocks_in_release_order;
  
  // fused blocks which is free, but it's phy_blocks are used by other block of my stream
  std::unordered_map<cudaStream_t, BlockEventOrderPool> fragmented_free_fused_blocks;
#endif

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

#ifdef MORE_POOL
  BlockPool ex1_blocks;
  BlockPool ex2_blocks;
#endif

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

#ifdef GMLAKE_ENABLE
  //active fused blocks
  ska::flat_hash_set<Block*> active_fused_blocks;
  
  //active fused blocks to be garbage collected
  ska::flat_hash_set<Block*> active_fused_blocks_to_gc;
#endif

#ifdef MEM_TWIN_REC
  SegmentManager segManager;
#endif

  // captures_underway tracks if a capture might be underway on any stream.
  // Most of the time it's zero, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  int captures_underway = 0;
  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding cuda events
  ska::flat_hash_map<
      cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;
#ifdef GMLAKE_ENABLE
  size_t total_fuse_size = 0;
  bool alloc_trace_record_context_ = false;
#endif
  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;
  std::vector<int> devices_with_peer_access_;

  bool set_fraction = false;

  bool record_history = false;
  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // Indicates that a current stream should be allocated to a pool
  // rather than the global memory.
  ska::flat_hash_map<cudaStream_t, MempoolId_t> stream_to_pool_map;
#ifdef GMLAKE_ENABLE
  // Maps a capturing stream to its assigned private pool,
  // in case we want multiple captures to share the same pool
  // ska::flat_hash_map<CaptureId_t, MempoolId_t> capture_to_pool_map; // replaced by stream_to_pool_map
#endif

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

 public:
#ifdef MORE_POOL
  DeviceCachingAllocator()
      : large_blocks(StatType::LARGE_POOL),
        small_blocks(StatType::SMALL_POOL),
        ex1_blocks(StatType::E1_POOL),
        ex2_blocks(StatType::E2_POOL),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }
#else
  DeviceCachingAllocator()
  #ifndef GMLAKE_ENABLE
      : large_blocks(/*small=*/false),
        small_blocks(/*small=*/true),
  #else
      : large_blocks(/*is_small=*/false),
        free_fused_blocks(/*is_small=*/false),
        small_blocks(/*is_small=*/true),
  #endif
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }
#endif

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

  bool isHistoryEnabled() {
    return record_history;
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    size_t allocated_pool_blocks = 0;

    for (Block* b : active_blocks) {
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }

        allocated_pool_blocks += 1;
      }
    }

    return allocated_pool_blocks == expected_live_allocations.size();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(int device, size_t orig_size, cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
// #ifndef GMLAKE_ENABLE
    size_t size = round_size(orig_size);
    auto context = maybeGatherContext(RecordContext::STATE);
// #else
//     CreateContextFn context_recorder = context_recorder_.load();
//     std::shared_ptr<Context> context =
//         context_recorder ? context_recorder() : nullptr;
// #endif

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway == 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events(context);
    }
    
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    bool block_found = false, move_defrag = false;

    if(c10::dtb::USE_DTR){
      if(!c10::dtb::move_defrag_flag[device]) {
        if(COST_FIRST_EVICT){
          auto *pm = c10::dtb::getDTBPoolManager();
          bool if_evict = pm->auto_evict(device, size);
        }
#ifdef MEM_TWIN_REC
        else{  // UNIFIED_EVICT
          trigger_free_memory_callbacks(params);
          // time_t begin = std::chrono::system_clock::now();
          auto if_evict = segManager.auto_evict(size, device, stream);
          // if (if_evict) {
          //   time_t end = std::chrono::system_clock::now();
          //   auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
          //   std::cout << "[sEvict]" << time_cost << "us" << std::endl;
          // }
          if(!if_evict&&UNIFIED_EVICT) {
            /// BUG[√]: 移动后会在auto_evict遍历segment报 segment fault（小心修改）
            segManager.auto_evict(size, device, stream, true); // gap size case
#ifdef DEFRAGMENT
  #ifdef DEBUG_MODE
            if(c10::dtb::record_move_defrag)
              std::cout << "[before move_for_defrag] " << c10::dtb::current_memory() / 1024 / 1024 << ", " << c10::dtb::reserved_memory() / 1024 / 1024 << "\n";
  #endif
            // bool if_mv = false;
            auto if_mv = segManager.move_for_defrag(size, device);

  #ifdef DEBUG_MODE
            if(c10::dtb::record_move_defrag)
              std::cout << "[after move_for_defrag] " << if_mv << " " << c10::dtb::current_memory() / 1024 / 1024 << ", " << c10::dtb::reserved_memory() / 1024 / 1024 << "\n";
  #endif
#endif
            // #ifdef DEBUG_MODE
            //   if(c10::dtb::record_move_defrag) {
            //     size_t before_mv = c10::dtb::current_memory(device);
            //     std::cout << "[evict fail] need:" << size 
            //               << ", curr-" << before_mv << std::endl;
            //   }
            // #endif
            // segManager.display_all_not_in_ap(device);
          }
        }
#endif
      } 
#ifdef MEM_TWIN_REC
      else {
        if(c10::dtb::move_defrag_flag[device]) {
          block_found = get_smaller_free_block(params, c10::dtb::move_defrag_max_size[device]) 
                        || (trigger_free_memory_callbacks(params) && get_smaller_free_block(params, c10::dtb::move_defrag_max_size[device]));
          move_defrag = true;
        }
      } 
#endif
    }


#ifdef MEM_TWIN_REC
    if(c10::dtb::USE_DTR){

    //   /// WIP: onsite defrag, not finished yet
    //   if(c10::dtb::defrag_flag[device]) {
    //     // here just for defragmentation on-site, because that this segment is surely to satisfy the memory needs
    //     Block* p = nullptr;
    //     for(auto& bit: segManager.get_defrag_seg()->blocks) {
    //       if(!bit->allocated) {
    //         p = bit;
    //         break;
    //       }
    //     }
    //     params.block = p;
    //     block_found = true;
    //   }

    }
#endif

    if (!block_found && !move_defrag) { // before may enter in defrag process
      // First, try to get a block from the existing pool.
      block_found =
          // Search pool
          get_free_block(params)
          // Trigger callbacks and retry search
  #ifndef GMLAKE_ENABLE
          || (trigger_free_memory_callbacks(params) && get_free_block(params));
  #else
          || (trigger_free_memory_callbacks(params) && get_free_block(params))
          || get_fused_fragmented_blocks(params, 0);
  #endif
    }

#ifdef DEFRAGMENT
    if (!block_found) {
      if(!c10::dtb::move_defrag_flag[device] && (c10::dtb::reserved_memory(device)+size)>c10::dtb::memory_budget && UNIFIED_EVICT) {
        auto if_mv = segManager.move_for_defrag(size, device);
        if(if_mv) {
          block_found = get_free_block(params);
        }
      }
    }
#endif

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
#ifndef GMLAKE_ENABLE
      block_found = alloc_block(params, false, context)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              alloc_block(params, false, context))
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway == 0) &&
              release_cached_blocks(context) &&
              alloc_block(params, true, context));
#else
      block_found = //alloc_block(params, false)
             realloc_block(params, false)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              realloc_block(params, false))
          || get_fused_fragmented_blocks(params, 1)
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway == 0) && release_cached_blocks(context) &&
              realloc_block(params, true))
          || get_fused_fragmented_blocks(params, 2);

      if (record_history && block_found) {
        record_trace(
            TraceEntry::SEGMENT_ALLOC,
            int64_t(params.block->ptr),
            params.block->size,
            params.stream(),
            context);
      }
#endif
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free = 0;
      size_t device_total = 0;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      std::string proc_info = reportProcessMemoryInfo(device);
      if(log_mem_events) {
        record_mem_events(
          TraceEntry::OOM,
          device_free,
          params.size());
      }
      if (record_history) {
        record_trace(
            TraceEntry::OOM,
            device_free,
            params.size(),
            params.stream(),
            std::move(context));
      }
      stats.num_ooms += 1;
#ifdef GMLAKE_ENABLE
      GMLAKE_INFO(" current memory info: device_total: %luMB, device_free: %luMB, request size: %luMB",
                                                                      device_total/(1024*1024), device_free/(1024*1024), size/(1024*1024));
      print_snapshot();
#endif
      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(c10::DeviceType::CUDA, static_cast<DeviceIndex>(device)));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();

      for (const auto& obs : observers_local) {
        obs(device,
            alloc_size,
            set_fraction ? allowed_memory_maximum : device_total,
            device_free);
      }

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "CUDA out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". GPU ",
          device,
          " has a total capacty of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          " is free. ",
          proc_info,
          "Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " If reserved but unallocated memory is large try setting max_split_size_mb to avoid"
          " fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF",
          "");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        std::move(params), orig_size, std::move(context), split_remainder);
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

#ifdef GMLAKE_ENABLE
    static const int vmmDefragment = ([]()->int{
      const char* env = getenv("vmmDefragment");
      if(env) return atoi(env);
      else return 1;
    })();
#endif

    const bool already_split = block->is_split();
    if (split_remainder) {
#ifdef GMLAKE_ENABLE
    if(pool->is_small || vmmDefragment <= 0 || (block->vmm_segment && !block->vmm_segment->fused)) {
#endif
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
      block->prev = remaining->prev;
      if (block->prev) {                // insert block ahead of remaining
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;   // remaining data pointer offset <size>, remaining means unallocated block
      remaining->size -= size;
#ifdef MEM_TWIN_REC
      // [TAG] split block and insert into it's SegmentTwin
      /**
       * remaining is orig block and in seg->blocks, block is a new block not in seg.
       * But the orig ptr which is in block2segment now belong to block.
       * Then remove record of orig ptr(to remaining) and redirect it to block.
      */
      if(UNIFIED_EVICT){
        auto *seg = segManager.get_segment_of_block(block->ptr, true /*remove*/);
        TORCH_INTERNAL_ASSERT(seg, "get a null segment!");
      #ifdef MEM_TWIN_DEBUG
        printf("[SPLIT] seg_members:%ld, add:%ld, reserve:%ld, new_block:%ld\n", seg->blocks.size(), 
          reinterpret_cast<uintptr_t>(remaining->ptr), reinterpret_cast<uintptr_t>((*(seg->blocks.begin()))->ptr),
          reinterpret_cast<uintptr_t>(block->ptr));
      #endif
        // now block and remaining are inserted in the seg->blocks
        segManager.add_block2segment(block, seg);              // and remaining->ptr is a new pointer to be added
        segManager.add_block2segment(remaining, seg);          // and remaining->ptr is a new pointer to be added
      }
#endif

#ifdef GMLAKE_ENABLE
      if(vmmDefragment > 0 && remaining->vmm_segment) {
              
          auto remaining_segment = remaining->vmm_segment->split(size);
          block->vmm_segment = std::move(remaining->vmm_segment);
          remaining->vmm_segment =  std::move(remaining_segment);
          
              
          size_t offset = 0;
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            phy_block->mapped_blocks[0].block = block;
            phy_block->mapped_blocks[0].offset = offset;
            phy_block->free = false;
            offset++;
          }
          block->vmm_segment->free_blocks = 0;
          block->vmm_segment->used_blocks = block->vmm_segment->phy_blocks.size();
              
              
          offset = 0;
          for(auto& phy_block : remaining->vmm_segment->phy_blocks) {
            phy_block->mapped_blocks[0].block = remaining;
            phy_block->mapped_blocks[0].offset = offset;
                  
            bool is_prev_free = phy_block->free;
            phy_block->free = true;
                  
            //neglect the the first block, since it is the remaining block
            for(int i=1; i<phy_block->mapped_blocks.size(); i++) {
              Block* other_block = phy_block->mapped_blocks[i].block;
                      
                      
              if(!is_prev_free) {
                other_block->vmm_segment->free_blocks++;
                        
                if(other_block->vmm_segment->fused) {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    if(other_block->stream == block->stream &&
                      fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                      fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                      free_fused_blocks.blocks.insert(other_block);
                      free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                    }
                  }
                }           
              }            
            }
                  
            offset++;
          }
          remaining->vmm_segment->free_blocks = remaining->vmm_segment->phy_blocks.size();
          remaining->vmm_segment->used_blocks = 0;
        }
#endif
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

#ifdef GMLAKE_ENABLE
      // if (context) {
      //   trimHistoryBefore(remaining, (char*)block->ptr + size);
      // }
#endif

      if (already_split && !block->expandable_segment_) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(
            stats.inactive_split_bytes,
            -static_cast<std::int64_t>(block->size),
            params.stat_types);
      } else if (!block->expandable_segment_) {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<std::int64_t>(remaining->size));
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }
////////////
#ifdef GMLAKE_ENABLE
    } /// Line:1844
      else if(vmmDefragment > 0 && block->vmm_segment) {
        size_t keep_blocks = size/kGranularity;
          
        if(block->vmm_segment->used_blocks > keep_blocks) {
          block->vmm_segment->free_blocks = block->vmm_segment->used_blocks - keep_blocks;
          block->vmm_segment->used_blocks = keep_blocks;
              
          for(size_t i=0; i<keep_blocks; i++) {
            if(block->vmm_segment->phy_blocks[i]->free) {
              GMLAKE_INFO(" warning for malloc fused blocks has free phy_block, something wrong happended");
              exit(-1);
            }
          }
              
          std::unordered_set<Block*> blocks2split;
          for(size_t i = keep_blocks; i < block->vmm_segment->phy_blocks.size(); i++) {
            auto& phy_block = block->vmm_segment->phy_blocks[i];
                  
            bool is_prev_free = phy_block->free;
            phy_block->free = true;
                  
            for(auto& block_segment : phy_block->mapped_blocks) {
              Block* other_block = block_segment.block;
              
              if(other_block == block) continue;
                      
              if(!is_prev_free) {
                other_block->vmm_segment->free_blocks++;
                if(other_block->vmm_segment->fused) {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    if(other_block->stream == block->stream &&
                      fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                      fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                      free_fused_blocks.blocks.insert(other_block);
                      free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                    }
                  }
                } else {
                  if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                    large_blocks.blocks.insert(other_block);
                                  
                    blocks2split.erase(other_block);
                    
                    if(other_block->is_split()) {
                      update_stat_array(stats.inactive_split, 1, params.stat_types);
                      update_stat_array(stats.inactive_split_bytes, other_block->size, params.stat_types);
                    }
                  } else {
                    if(blocks2split.count(other_block) == 0) {
                      blocks2split.insert(other_block);
                    }
                  }
                } 
              }
            }
          }
    
          for(auto& block2split : blocks2split) {
            if(block2split->vmm_segment->fused || 
                block2split->vmm_segment->free_blocks == 0 || 
                block2split->vmm_segment->free_blocks == block2split->vmm_segment->phy_blocks.size()) continue;
                  
                  
            if(active_blocks.count(block2split)) {
              block2split->allocated = false;
              active_blocks.erase(block2split);
                    
              update_stat_array(stats.active, -1, params.stat_types);
              update_stat_array(stats.active_bytes, -block2split->size, params.stat_types);
            }
                  
                  
            bool block_free = block2split->vmm_segment->phy_blocks[0]->free;          
            size_t last_offset = 0;
            Block* prev_block = block2split->prev;
                  
            auto phy_blocks = block2split->vmm_segment->phy_blocks;
            auto vmm_segment = std::move(block2split->vmm_segment);
                  
            for(size_t i=1; i<=phy_blocks.size(); i++) {
                      
              if( i == phy_blocks.size() || block_free != phy_blocks[i]->free ) {
                size_t block_size = (i - last_offset)*kGranularity;
                          
                char* block_ptr = (char*)block2split->ptr + last_offset*kGranularity;
                Block* split_block = new Block(device, stream, block_size, pool, block_ptr);
                          
                          
                split_block->prev = prev_block;
                if(prev_block) prev_block->next = split_block;
            
                split_block->self_last_event = block2split->self_last_event;
                          
                          
                if(i < phy_blocks.size()) {
                  auto remaining_segment = vmm_segment->split(block_size);
                  split_block->vmm_segment = std::move(vmm_segment);
                  vmm_segment = std::move(remaining_segment);
                } else {
                  split_block->vmm_segment = std::move(vmm_segment);
                }
                          
                          
                size_t offset = 0;
                for(auto& phy_block : split_block->vmm_segment->phy_blocks) {
                  phy_block->mapped_blocks[0].block = split_block;
                  phy_block->mapped_blocks[0].offset = offset;
                  offset++;
                }
                          
            
                if(block_free) {
                  split_block->vmm_segment->free_blocks = split_block->vmm_segment->phy_blocks.size();
                  split_block->vmm_segment->used_blocks = 0;
              
                  large_blocks.blocks.insert(split_block);
              
                  update_stat_array(stats.inactive_split, 1, params.stat_types);
                  update_stat_array(stats.inactive_split_bytes, split_block->size, params.stat_types);
                } else {
                  split_block->vmm_segment->free_blocks = 0;
                  split_block->vmm_segment->used_blocks = 0;
                              
                  split_block->allocated = true;
                  active_blocks.insert(split_block);
                              
                              
                  update_stat_array(stats.active, 1, params.stat_types);
                  update_stat_array(stats.active_bytes, split_block->size, params.stat_types);
                }
                    

                if(i < phy_blocks.size()) {
                  block_free = phy_blocks[i]->free;
                }
                last_offset = i;
                prev_block = split_block;
              }
            }
                  
                  
            if(prev_block) {
              prev_block->next = block2split->next;
            }
                  
            if(block2split->next) {
              block2split->next->prev = prev_block;
            }
                  
                  
            delete block2split;
          }
        }
      }

      // if (record_history) {
      //   trimHistoryBefore(remaining, (char*)block->ptr + size);
      // }

#endif
    } else if (already_split && !block->expandable_segment_) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(
            stats.inactive_split_bytes[stat_type],
            -static_cast<std::int64_t>(block->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;

#ifdef GMLAKE_ENABLE
    block->actual_size = size;
#endif
    if(log_mem_events) {
      record_mem_events(
        TraceEntry::ALLOC,
        int64_t(block->ptr),
        orig_size);
    }
    if (record_history) {
#ifdef GMLAKE_ENABLE
      // trimHistoryBefore(block, (char*)block->ptr + size);
      block->history = std::make_unique<HistoryChain>(HistoryChain{
          History{block->ptr, orig_size, std::move(context)},
          std::move(block->history)});
      if (!block->history_last) {
          block->history_last = block->history.get();
      }
#endif
      block->context_when_allocated = std::move(context);
      record_trace(
          TraceEntry::ALLOC,
          int64_t(block->ptr),
          orig_size,
          block->stream,
          block->context_when_allocated);
    }

#ifdef GMLAKE_ENABLE
    bool inserted = false;
    if(block->vmm_segment && block->vmm_segment->fused) {
        active_fused_blocks.insert(block);
    } else {
        inserted = active_blocks.insert(block).second;
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }
#else
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
#endif

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(
          stats.allocated_bytes[stat_type],
          static_cast<std::int64_t>(block->size));
#ifndef GMLAKE_ENABLE // TAG: Maybe wrong
      update_stat(stats.active[stat_type], 1);
      update_stat(
          stats.active_bytes[stat_type],
          static_cast<std::int64_t>(block->size));
#endif
      update_stat(
          stats.requested_bytes[stat_type],
          static_cast<std::int64_t>(block->requested_size));
#ifdef GMLAKE_ENABLE
      // if (inserted)
      // {
          update_stat(stats.active[stat_type], 1);
          update_stat(stats.active_bytes[stat_type], block->size);
      // }
#endif
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  void free(Block* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(
          stats.allocated_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });
    if(log_mem_events) {
      record_mem_events(
        TraceEntry::FREE_REQUESTED,
        int64_t(block->ptr),
        block->requested_size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::FREE_REQUESTED,
          int64_t(block->ptr),
          block->requested_size,
          block->stream,
          context ? context : block->context_when_allocated);
    }
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(captures_underway)) {
        // It's forbidden to cudaEventQuery an event recorded during CUDA graph
        // capture. We conservatively defer recording end-of-life events until
        // the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
#ifndef GMLAKE_ENABLE
      free_block(block, context);
#else
      insert_free_event_into_alloc_stream(block);   
      update_block(block, context);
#endif
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

#ifdef GMLAKE_ENABLE
  void update_block(Block* block, const std::shared_ptr<GatheredContext>& context) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    bool flag = false;
      
    std::unordered_set<Block*> blocks2free;
    if(block->vmm_segment) {
          
      for(size_t i=0; i<block->vmm_segment->phy_blocks.size(); i++) {
              
        if(i < block->vmm_segment->used_blocks) {
          auto& phy_block = block->vmm_segment->phy_blocks[i];
              
          bool is_prev_free = phy_block->free;
          if(!is_prev_free) {
            block->vmm_segment->free_blocks++;
            phy_block->free = true;
          } else {
            GMLAKE_INFO(" warning used blocks is free");
            exit(-1);
          }
                  
          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                      
            if(other_block == block) continue;
                      
            if(!is_prev_free) {
              other_block->vmm_segment->free_blocks++;
                          
              if(other_block->vmm_segment->fused) {
                if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                  if(other_block->stream == block->stream &&
                    fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                    fragmented_free_fused_blocks[other_block->stream].erase(other_block);
                                      
                    free_fused_blocks.blocks.insert(other_block);
                    free_fused_blocks_in_release_order[other_block->stream].insert(other_block);
                  }
                }
              } else {
                if(!other_block->self_last_event ||
                    other_block->self_last_event->event_id < block->self_last_event->event_id) {
                    other_block->self_last_event = block->self_last_event;
                }
                              
                if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                  blocks2free.insert(other_block);
                }
              }
                          
            }
          }
        }
      }
          
      block->vmm_segment->used_blocks = 0;
          
    }
      
      
    if(block->vmm_segment && block->vmm_segment->fused) {
      if(active_fused_blocks_to_gc.count(block) == 0) {
        if(block->vmm_segment->free_blocks == block->vmm_segment->phy_blocks.size()) {
          if(fragmented_free_fused_blocks[block->stream].blocks.count(block)) {
            fragmented_free_fused_blocks[block->stream].erase(block);
          }
                  
          free_fused_blocks.blocks.insert(block);
          free_fused_blocks_in_release_order[block->stream].insert(block);
        } else {
          fragmented_free_fused_blocks[block->stream].insert(block);
        }
      }

          
      if(active_fused_blocks.count(block)) {
        block->allocated = false;
        active_fused_blocks.erase(block);
        flag = true;
        size_t requested_size = block->requested_size;
        StatTypes stat_types = get_stat_types_for_pool(*block->pool);
        for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
          update_stat(
            stats.requested_bytes[stat_type],
            -static_cast<std::int64_t>(requested_size));
        });
              
        if(active_fused_blocks_to_gc.count(block)) {
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            int i = 0;
            for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
              if(phy_block->mapped_blocks[j].block != block) {
                if(i != j) {
                  phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                }
                              
                i++;
              }
            }
            phy_block->mapped_blocks.resize(i);
          }
                  
          active_fused_blocks_to_gc.erase(block);
          delete block;
        }
      }
    } else {
      free_block(block, context, flag);
    }
      
      
    for(auto& block2free : blocks2free) {
          
      block2free->allocated = false;
      free_block(block2free, context, flag);
    }
      
  }
#endif

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(
        !block->expandable_segment_,
        "Tensors allocated with expandable_segments:True cannot be shared between processes. Consider using expandable_segments:False in data loading workers via torch.cuda.memory._set_allocator_settings('expandable_segments:False')");
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free = 0;
    size_t device_total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context);
#ifdef GMLAKE_ENABLE
    size_t garbage_size = garbage_collect_fused_blocks(2, 0);
    total_fuse_size -= garbage_size;
	
	  GMLAKE_INFO(" garbage_collect_fused_blocks() return %luMB garbage memory", garbage_size/(1024*1024));
#endif
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes = 0;
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes));
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
#ifdef MORE_POOL
    cache_info_aux(ex1_blocks, largest);
    cache_info_aux(ex2_blocks, largest);
#endif
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  void logPtrInfo(std::vector<void*> ptrs) {
    auto *pm = c10::dtb::getDTBPoolManager();
#ifdef MEM_TWIN_REC
    if(ptrs.size()>0) {
      DTRLogMemEvents("SINGLE_COMM_BEGIN", 0, 0);
      for(const auto& ptr: ptrs) {
        // auto res = pm->get_ap_by_ptr(ptr);
        auto seg = segManager.get_segment_of_block(ptr);
        if(seg){
          uintptr_t addr = 0;
          for(auto& bit: seg->blocks) {
            addr = reinterpret_cast<uintptr_t>(bit->ptr);
            break;
          }
          DTRLogMemSegmentStat("COMM_LOG", seg->total_size, addr, seg->blocks.size());
        }
      }
      DTRLogMemEvents("SINGLE_COMM_END", 0, 0);
    }
#endif
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /* Checkpoint the state of a private pool necessary to return it to its
   * current state */
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_blocks =
          get_private_pool_head_blocks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    std::unordered_map<void*, Block*> orig_ptrs_to_blocks;

    auto pool_blocks = get_private_pool_head_blocks(private_pool);

    std::vector<Block*> head_blocks;
    for (Block* block : pool_blocks) {
      if (block->prev == nullptr) {
        head_blocks.push_back(block);
      }
    }

    for (Block* block : head_blocks) {
      Block* curr = block;

      while (curr) {
        // When we free a block, its pointer should never change
        // only its adjacent blocks, so free, then look at pointer
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "Events should have synchronized when setting checkpointed block");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    for (Block* b : get_private_pool_head_blocks(private_pool)) {
      Block* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // checkpoint the state of an allocation that may have been
  // split into multiple blocks
  void setSegmentStateToCheckpoint(
      Block* block,
      SegmentState& segment,
      std::shared_ptr<GatheredContext> context,
      RestoreResult& rr) {
    Block* curr_block = block;
    Block* last_block = block;

    TORCH_INTERNAL_ASSERT(block->pool);
    BlockPool& pool = *block->pool;
    const auto segment_len = segment.blocks.size();

    // allocate all blocks in the segment
    for (size_t i = 0; i < segment_len; ++i) {
      auto& block_state = segment.blocks.at(i);
      AllocParams params(
          block_state.device,
          block_state.size,
          block_state.stream,
          &pool,
          block_state.size,
          stats);
      pool.blocks.erase(curr_block);
      params.block = curr_block;
      params.stat_types = get_stat_types_for_pool(pool);

      // splitting a block depends on `max_split_size`, which may have changed
      // between whe checkpoint was taken and now, so we make sure to recreate
      // the behavior from the checkpoint.
      bool split = (i + 1) < segment.blocks.size();

      // curr_block will become next pointer if it is split, so reassign with
      // the returned value
      curr_block = alloc_found_block(
          std::move(params), block_state.size, context, split);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->size == block_state.size);

      last_block = curr_block;
      curr_block = curr_block->next;

      TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)));
    }

    while (last_block->prev) {
      last_block = last_block->prev;
    }

    // free blocks that are not allocated in the checkpoint
    curr_block = last_block;

    for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
      auto& block_state = segment.blocks.at(i);
      TORCH_INTERNAL_ASSERT(curr_block != nullptr);

      if (block_state.allocated) {
        rr.allocations_created.push_back(curr_block);
        continue;
      }

      free(curr_block);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->allocated == block_state.allocated);
      TORCH_CHECK(curr_block->size == block_state.size);
    }
  }

  /**
   * Note [Checkpointing PrivatePoolState]
   *
   * Refer above to Note [Interaction with CUDA graph capture]. Allocations made
   * during graph capture are made from a separate private pool. During graph
   * capture allocations behave as usual. During graph replay the allocator
   * state does not change even as new tensors are created. The private pool
   * will not free its blocks to the main caching allocator until cuda graph use
   * is finished to prevent an allocation from eager clobbering the memory from
   * a live but unaccounted for tensor that was created during replay.
   *
   * `make_graphed_callables`, a series of separate callables chained in
   * successive cuda graphs, can share a memory pool because after a cuda graph
   * recording the allocations in the shared private pool exactly reflect the
   * tensors that are allocated.
   *
   * We would like to extend callable chaining to support a graphed callable
   * tree. In this scenario, we have a tree of callable chains which will be
   * captured with cuda graphs. In the diagram below, we have a tree with four
   * callables, A, B, C, and D. Suppose we have captured, and subsequently
   * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
   * would now like to record D. At this point the private pool will not reflect
   * any of the live tensors created during graph replay. Allocations made
   * during a new recording with the pool could overwrite those live tensors.
   *
   * In order to record a new graph capture after replaying prior callables in
   * the tree, we need the allocator to reflect the state of the live tensors.
   * We checkpoint the state of the private pool after each recording, and then
   * reapply it when we are starting a new recording chain. Additionally, we
   * must free the allocations for any tensors that died between the end of our
   * previous graph replaying and our new recording. All of the allocated
   * segments that existed in the checkpointed state must still exist in the
   * pool. There may also exist new allocated blocks.
   * (TODO : link note [live tensors between iterations] when it exists). For
   * every block that is currently allocated but no allocated in the snapshot,
   * we will return a pointer to their block.
   *.
   *
   *
   *  ---------------> A ---------------> B ---------------> C
   *                                      |
   *                                      |
   *                                      |
   *                                      |
   *                                      ╰ ---------------> D
   */
  RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
    // To reset the caching allocator state we will
    // - Free all the blocks currently allocated to the pool (see [live tensors
    // between iterations])
    // - Allocate all the blocks in a checkpointed segment, whether they are
    // live or not
    // - Free the blocks in a checkpointed segment which are not live
    // This could be optimized, but it nicely reuses exiting apis, and this
    // is not on the hot path.

    // following `done outside the lock because we don't know what locks the
    // recorder needs to have...`

    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::STATE);

    std::lock_guard<std::recursive_mutex> lock(mutex);

    RestoreResult rr;

    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    PrivatePool* private_pool = pool->second.get();

    freeBlocksAllocatedToPool(private_pool, rr);

    std::unordered_map<void*, Block*> ptrs_to_blocks;
    // at this point, all of the blocks should be free, so they will all be in
    // the block set
    for (Block* block : private_pool->small_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }
    for (Block* block : private_pool->large_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }

    for (auto& segment : pps.segments) {
      auto ptr = segment.blocks.at(0).ptr;
      TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr)
      auto block = ptrs_to_blocks[ptr];

      setSegmentStateToCheckpoint(block, segment, context, rr);
    }
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::unordered_map<PrivatePool*, MempoolId_t> pool_to_id;
    pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
    for (const auto& pair : graph_pools) {
      pool_to_id[pair.second.get()] = pair.first;
    }
    for (const auto& pair : graph_pools_freeable) {
      pool_to_id[pair.second] = pair.first;
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();
    for (const Block* const head_block : all_blocks) {
      // For expandable segments, we report one segment for each continguous
      // mapped range of memory
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
#ifdef MORE_POOL
      segment_info.is_large = (head_block->pool->pool_type!=StatType::SMALL_POOL);
      segment_info.segment_type = head_block->pool->pool_type;
#else
      segment_info.is_large = (!head_block->pool->is_small);
#endif
      segment_info.is_expandable = head_block->expandable_segment_;
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;
      auto mempool_id = pool_to_id.find(head_block->pool->owner_PrivatePool);
      if (mempool_id != pool_to_id.end()) {
        segment_info.owner_private_pool_id = mempool_id->second;
      }

      const Block* block = head_block;
      while (block != nullptr && block->mapped) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        block_info.context_when_allocated = block->context_when_allocated;
        
// #ifdef GMLAKE_ENABLE
//         HistoryChain* h = block->history.get();
//         while (h) {
//           block_info.history.push_back(h->h);
//           h = h->next.get();
//         }
// #endif
        block = block->next;
      }
      // std::cout << "snapshot - for: stream-" << head_block->stream << ", blocks-" << count << "\n";
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });
    if(log_mem_events) {
      record_mem_events(
        TraceEntry::SNAPSHOT,
        0,
        total_active);
    }
    if (record_history) {
      record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, nullptr);
    }
    return result;
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() + alloc_trace_next,
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() + alloc_trace_next);
    return result;
  }

#ifdef GMLAKE_ENABLE
  void print_snapshot()
  {
    auto memory_snapshot = snapshot();
      
    for(auto& segment_info : memory_snapshot) {
      if(segment_info.is_large) {
        GMLAKE_INFO(" segment: %p, size: %luMB", (void*)segment_info.address, segment_info.total_size/(1024*1024));
                    
        for(auto& block_info : segment_info.blocks) {
          GMLAKE_INFO(" %s %s block, size: %luMB", 
                     (block_info.allocated? "allocated" : "unallocated"), 
                     (block_info.active? "active" : "inactive"),
                     block_info.size/(1024*1024) );
        }
      }
    }
  }
#endif

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

#ifdef MORE_POOL
  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }
#else
  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
#ifndef GMLAKE_ENABLE
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
#else
        size_t block_round_size = kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
        if(block_round_size > kSmallSize) //if block will alloc from large_blocks, round to 2M
        {
            block_round_size = kGranularity * ((size + kGranularity - 1) / kGranularity);
        }
        return block_round_size;
#endif
      }
    }
  }
#endif

  // See Note [Interaction with CUDA graph capture]

  // Called by CUDAGraph::capture_begin
  void beginAllocateStreamToPool(cudaStream_t stream, MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway++;
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool. Make a new pool for
      // this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // mempool_id references an existing pool, which the current capture will
      // share. Check this pool is live (at least one other capture already
      // references it).
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }

    // Maps this stream to mempool_id and makes sure this graph_id wasn't
    // somehow assigned a mempool_id already. Keeps essential effect (insert)
    // out of macro.
    bool inserted = stream_to_pool_map.insert({stream, mempool_id}).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }


  // Called by CUDAGraph::capture_end
  void endAllocateStreamToPool(cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway--;
    auto it = stream_to_pool_map.find(stream);
    TORCH_INTERNAL_ASSERT(it != stream_to_pool_map.end());
    stream_to_pool_map.erase(it);
  }

  // Called by CUDAGraph::reset
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated cudaGraphExec_t has been destroyed. We can't blindly
    // delete and cudaFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_blocks it may now
    // cudaFree blocks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  void addPeerAccess(int dev_to_access) {
    if (std::find(
            devices_with_peer_access_.begin(),
            devices_with_peer_access_.end(),
            dev_to_access) != devices_with_peer_access_.end()) {
      return;
    }
    devices_with_peer_access_.push_back(dev_to_access);
    for (auto& es : expandable_segments_) {
      es->addPeer(dev_to_access);
    }
  }

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

#ifdef MEM_TWIN_REC
  void getSegmentTwins() {
    DTRLogger::logger().log("{\"TYPE\": \"TAG\", \"VALUE\": \"BEGIN\"}");
    // SegmentTwin* seg = segManager.getMinSegment();
    // // for(auto& ele: segManager.blocks2segment){     // 不能遍历blocks2segment，这个map的个数其实是block的个数
    // while(seg) {
    //   // uintptr_t ptr = reinterpret_cast<uintptr_t>(ele.first);
    //   auto it = seg->blocks.begin();
    //   uintptr_t ptr;
    //   if(it!=seg->blocks.end())
    //     ptr = reinterpret_cast<uintptr_t>(it.current->value->ptr);
    //   else
    //     ptr = 0;
    //   size_t segment_size = seg->total_size;
    //   size_t blocks_num = seg->blocks.size();
    //   auto last_time = seg->last_change_time;
    //   auto duration = last_time.time_since_epoch();
    //   size_t millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();  // timestamp
    //   DTRLogSegmentsStats(segment_size, blocks_num, millis, ptr);
    //   seg = seg->next;
    // }
    segManager.display_segments();
    DTRLogger::logger().log("{\"TYPE\": \"TAG\", \"VALUE\": \"END\"}");
  }
#endif

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
#ifdef MORE_POOL
    blocks.insert(
        blocks.end(), ex1_blocks.blocks.begin(), ex1_blocks.blocks.end());
    blocks.insert(
        blocks.end(), ex2_blocks.blocks.begin(), ex2_blocks.blocks.end());
#endif
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const {
    std::vector<Block*> blocks;
    for (Block* b : active_blocks) {
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    for (Block* b : pool->small_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }
    for (Block* b : pool->large_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    return blocks;
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Block* find_expandable_block(
      int device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size) {
    Block key(device, stream, 0);

    auto allocatable = [](Block* b) {
      return b && !b->allocated && b->event_count == 0 &&
          b->stream_uses.empty();
    };
    auto has_available_address_space = [&](Block* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream;
         ++it) {
      Block* c = *it;
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
#ifdef MORE_POOL
    size_t segment_size;
    if(pool->pool_type==StatType::SMALL_POOL) segment_size = kSmallBuffer;
    else if(pool->pool_type==StatType::E1_POOL) segment_size = kE1Buffer;
    else if(pool->pool_type==StatType::E2_POOL) segment_size = kE2Buffer;
    else segment_size = kLargeBuffer;
#else
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
#endif
    expandable_segments_.emplace_back(new ExpandableSegment(
        device, stream, segment_size, devices_with_peer_access_));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(
      Block* to_map,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated); // unmapped blocks should not keep
                                          // history
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(
        mapped_range.ptr == to_map->ptr && mapped_range.size >= size);

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block* remaining = new Block(
          to_map->device,
          to_map->stream,
          to_map->size - mapped_range.size,
          &pool,
          static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.blocks.insert(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });
    if(log_mem_events) {
      record_mem_events(
        TraceEntry::SEGMENT_MAP,
        int64_t(mapped_range.ptr),
        mapped_range.size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::SEGMENT_MAP,
          int64_t(mapped_range.ptr),
          mapped_range.size,
          to_map->stream,
          ctx);
      if (!to_map->prev && !to_map->context_when_segment_allocated) {
        to_map->context_when_segment_allocated = ctx;
      }
    }

    return true;
  }

  Block* try_allocate_expandable_block(
      int device,
      cudaStream_t stream,
      BlockPool* pool,
      size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    Block* candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(
              new_candidate, std::min(remaining, candidate->next->size), ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(
      Block* block,
      const std::shared_ptr<GatheredContext>& context, bool flag=false) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());
    if(log_mem_events) {
      record_mem_events(
          TraceEntry::FREE_COMPLETED,
          int64_t(block->ptr),
          block->requested_size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::FREE_COMPLETED,
          int64_t(block->ptr),
          block->requested_size,
          block->stream,
          context ? context : block->context_when_allocated);
    }
    block->context_when_allocated = nullptr;
#ifdef GMLAKE_ENABLE
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();

#endif
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

#ifdef GMLAKE_ENABLE
    if(vmmDefragment > 0 && block->vmm_segment/*!pool.is_small*/) {
      block->vmm_segment->free_blocks = block->vmm_segment->phy_blocks.size();
      block->vmm_segment->used_blocks = 0;

    }
#endif

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);

    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segements from
      // inactive_split
      if (!block->expandable_segment_) {
        update_stat(
            stats.inactive_split[stat_type], net_change_inactive_split_blocks);
        update_stat(
            stats.inactive_split_bytes[stat_type],
            net_change_inactive_split_size);
      }
      update_stat(stats.active[stat_type], -1);
      update_stat(
          stats.active_bytes[stat_type],
          -static_cast<std::int64_t>(original_block_size));
// TAG: Maybe wrong
      if (!flag) {
        update_stat(
            stats.requested_bytes[stat_type],
            -static_cast<std::int64_t>(requested_size));
      }
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst] --> [  dst  ]
#ifdef MEM_TWIN_REC
      if(UNIFIED_EVICT){
        auto *seg = segManager.get_segment_of_block(dst->ptr, true /* remove */);    // record of current dst->ptr is deleted
        TORCH_INTERNAL_ASSERT(seg!=nullptr);
      #ifdef MEM_TWIN_DEBUG
        printf("[MERGE src dst] seg_members:%ld, erase:%ld, reserve:%ld\n", seg->blocks.size(), reinterpret_cast<uintptr_t>(src->ptr), reinterpret_cast<uintptr_t>((*(seg->blocks.begin()))->ptr));
      #endif 
        seg->erase(src);  // release block src
      #ifdef MEM_TWIN_DEBUG
        printf("[AFTER MERGE src dst] seg_members:%ld, reserve:%ld\n", seg->blocks.size(), reinterpret_cast<uintptr_t>((*(seg->blocks.begin()))->ptr));
      #endif
        TORCH_INTERNAL_ASSERT(!seg->empty(), "erase get a empty segment!");
      }
#endif
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);

#ifdef GMLAKE_ENABLE
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        src->history_last->next = std::move(dst->history);
        dst->history = std::move(src->history);
      }
      src->history_last = nullptr;
#endif
    } else { // [dest src] --> [  dst  ]
#ifdef MEM_TWIN_REC
      if(UNIFIED_EVICT){
        auto *seg = segManager.get_segment_of_block(src->ptr, true /* remove */);
        TORCH_INTERNAL_ASSERT(seg!=nullptr);
        size_t before_size = seg->blocks.size();
      #ifdef MEM_TWIN_DEBUG
        printf("[MERGE dst src] seg_members:%ld, erase:%ld\n", seg->blocks.size(), reinterpret_cast<uintptr_t>(src->ptr));
      #endif
        seg->erase(src);
      #ifndef MEM_TWIN_DEBUG
        TORCH_INTERNAL_ASSERT(!seg->empty(), "erase get a empty segment!");
      #else
        if(seg->empty()){
          printf("dst:%ld(%ld), src:%ld(%ld), before_erase_seg_size:%ld(%ld)\n", 
            reinterpret_cast<uintptr_t>(dst->ptr), dst->size, 
            reinterpret_cast<uintptr_t>(src->ptr), src->size,
            before_size, seg->total_size);
          TORCH_INTERNAL_ASSERT(!seg->empty(), "erase get a empty segment!");
        }
      #endif
      }
#endif
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
#ifdef GMLAKE_ENABLE
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        dst->history_last->next = std::move(src->history);
        dst->history_last = src->history_last;
      }
      src->history_last = nullptr;
#endif
    }
#ifdef GMLAKE_ENABLE    
    std::shared_ptr<BlockEvent> current_self_last_event = src->self_last_event;
    if(!current_self_last_event || (dst->self_last_event && dst->self_last_event->event_id > current_self_last_event->event_id)) {
      current_self_last_event = dst->self_last_event;
    }
#endif
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
#ifdef GMLAKE_ENABLE  
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();


    if(vmmDefragment > 0 && dst->vmm_segment) {
      bool ret = dst->vmm_segment->remerge(*(src->vmm_segment));
      if(!ret) {
        GMLAKE_INFO(" merge block %p, ptr %p of size %fMB into block %p, ptr %p of size %fMB failed", 
                    src, src->ptr, src->size/(1024.f*1024.f), dst, dst->ptr, dst->size/(1024.f*1024.f));
      }
      
      size_t offset = 0;
      for(auto& phy_block : dst->vmm_segment->phy_blocks) {
          phy_block->mapped_blocks[0].block = dst;
          phy_block->mapped_blocks[0].offset = offset;
          offset++;
      }
    }
#endif

    delete src;

    return subsumed_size;
  }

#ifndef MORE_POOL
  BlockPool& get_pool(size_t size, cudaStream_t stream) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only > 0 if some thread has begun and not yet ended a
    // capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway)) {
// #ifdef GMLAKE_ENABLE
//       CaptureId_t id;
//       cudaStreamCaptureStatus status;
//       C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &id));
//       if (status != cudaStreamCaptureStatus::cudaStreamCaptureStatusNone) {
//         TORCH_INTERNAL_ASSERT(
//             status !=
//             cudaStreamCaptureStatus::cudaStreamCaptureStatusInvalidated);
//         // Retrieves the private pool assigned to this capture.
//         auto it0 = capture_to_pool_map.find(id);
//         TORCH_INTERNAL_ASSERT(it0 != capture_to_pool_map.end());
//         auto it1 = graph_pools.find(it0->second);
//         TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
//         if (size <= kSmallSize) {
//           return it1->second->small_blocks;
//         } else {
//           return it1->second->large_blocks;
//         }
//       }
// #else
      auto it0 = stream_to_pool_map.find(stream);
      if (it0 != stream_to_pool_map.end()) {
        auto it1 = graph_pools.find(it0->second);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
        if (size <= kSmallSize) {
          return it1->second->small_blocks;
        } else {
          return it1->second->large_blocks;
        }
      }
// #endif
    }
#endif
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }
#else
  BlockPool& get_pool(size_t size, cudaStream_t stream) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only > 0 if some thread has begun and not yet ended a
    // capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway)) {
      auto it0 = stream_to_pool_map.find(stream);
      if (it0 != stream_to_pool_map.end()) {
        auto it1 = graph_pools.find(it0->second);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
        if (size <= kSmallSize) {
          return it1->second->small_blocks;
        } else {
          return it1->second->large_blocks;
        }
      }
    }
#endif
    // TODO: change get_block logic, choosed by if lock this value
    // if(UNIFIED_EVICT){
    //   if(c10::dtb::store_in_special_pool[current_device()])
    //   {
    //     return ex1_blocks;
    //   }
    // }
    // if(UNIFIED_EVICT) {
    //   if(!c10::dtb::in_runtime_record[c10::cuda::current_device()]) {
    //     std::cout << "[get_ex pool] in runtime record with: " << size /1024/1024 << "MB.\n";
    //     return ex1_blocks;
    //   }
    // }
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
    // if (size <= kSmallSize) {
    //   return small_blocks;
    // } else if (size <= kE1Size){
    //   return ex1_blocks;
    // } else if (size <= kE2Size){
    //   return ex2_blocks;
    // } else {
    //   return large_blocks;
    // }
  }
#endif


  StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
#ifdef MORE_POOL
    stat_types[static_cast<size_t>(
        pool.pool_type)] = true;
#else
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
#endif
    return stat_types;
  }

#ifdef MORE_POOL  /// TODO: 这里是否有必要对拆分大小做限制？
  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->pool_type==StatType::SMALL_POOL ||
        CachingAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }
#else
  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        CachingAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }
#endif

#ifndef MORE_POOL
  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {                                           // 1MB以下padding到2MB
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {                                 // 1-10MB内padding到20MB
      return kLargeBuffer;
    } else {                                                            // 否则按2MB的倍数分配
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }
#else
  // segment allocation size
  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {                                           // 1MB以下padding到2MB
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {                                 // 1-10MB内padding到20MB
      return kLargeBuffer;
    } else {                                                            // 否则按2MB的倍数分配
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
    // if (size <= kSmallSize) {                                           // 1MB以下padding到2MB
    //   return kSmallBuffer;
    // } 
    // else if (size < kE1Size) { // kMinE1Alloc       // small-E1    // TODO: 适配这里的逻辑到锁定张量
    //   return kE1Buffer;
    // } 
    // else if (size < kE2Size) {  // kMinE2Alloc        // E1-E2
    //   // return kE2Buffer;
    //   return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    // } 
    // else if (size < kMinLargeAlloc) {                 // E2-Large
    //   return kE2Buffer;
    // } 
    // else {                                                           // 否则按2MB的倍数分配
    //   return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    // }
  }
#endif

  bool get_free_block(AllocParams& p) {
#ifdef GMLAKE_ENABLE
    static const int vmmDefragment = []() {
        const char* env = getenv("vmmDefragment");
        return env ? atoi(env) : 1;
    }();

    static const double reuseLimit = []() {
        const char* env = getenv("reuseLimit");
        return env ? atof(env) : 10.0;
    }();
    
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;  
#endif
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    /// below codes are for the multi stream overlap
    // if(!pool.blocks.empty()) {
    //   auto it = pool.blocks.begin();
    //   if(p.stream()!=(*it)->stream) {
    //     printf("which stream is default? %d %d\n", p.stream()==cudaStreamDefault ? 1 : 0, (*it)->stream==cudaStreamDefault ? 1 : 0);
    //     p.search_key.stream = (*it)->stream;
    //   }
    // }
    // if(c10::dtb::getStreamLabel(p.stream())!=-1)
    //   p.search_key.stream = cudaStreamDefault;

    auto it = pool.blocks.lower_bound(&p.search_key);     // return first ge search_key container
#ifdef GMLAKE_ENABLE
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      if(vmmDefragment > 0 && !pool.is_small) {
        auto block_it = free_fused_blocks.blocks.lower_bound(&p.search_key);
        if (block_it == free_fused_blocks.blocks.end() 
            || (*block_it)->stream != p.stream() 
            || (*block_it)->size > (p.search_key.size*reuseLimit))
        {
          return false;
        }
                              
            
        p.block = *block_it;
            
            
        size_t keep_blocks = p.search_key.size/kGranularity;
           
        std::unordered_set<Block*> blocks2split;
        for(size_t i=0; i < keep_blocks; i++) {
          auto& phy_block = p.block->vmm_segment->phy_blocks[i];
                
          if(!phy_block->free) {
            GMLAKE_INFO(" warning for fused blocks not free, something wrong happended");
            exit(-1);
          }
                
          phy_block->free = false;

          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                    
            if(other_block == p.block) continue;
                    
            if(other_block->vmm_segment->fused) {
              if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
                free_fused_blocks.blocks.count(other_block)) {
                  free_fused_blocks.blocks.erase(other_block);
                  free_fused_blocks_in_release_order[other_block->stream].erase(other_block);

                            
                  fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              } else if(active_fused_blocks.count(other_block) == 0) {
                if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                  fragmented_free_fused_blocks[other_block->stream].insert(other_block);
                }
              }
                        
                        
              other_block->vmm_segment->free_blocks--;
            } else {
              if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size()) {
                if(large_blocks.blocks.count(other_block)) {
                  large_blocks.blocks.erase(other_block);
                                 
                  blocks2split.insert(other_block);
               
                  if(other_block->is_split()) {
                    net_change_inactive_split_blocks -= 1;
                    net_change_inactive_split_size -= other_block->size;
                  }
                }
              }
                        
                        
              other_block->vmm_segment->free_blocks--;
                        
                        
              if(other_block->vmm_segment->free_blocks == 0) {
                blocks2split.erase(other_block);
                            
                other_block->allocated = true;
                active_blocks.insert(other_block);
                            
                            
                update_stat_array(stats.active, 1, p.stat_types);
                update_stat_array(stats.active_bytes, other_block->size, p.stat_types);
              }
            }
          }
        }
            
            
        for(auto& block2split : blocks2split) {      
          if(block2split->vmm_segment->fused || 
            block2split->vmm_segment->free_blocks == 0 || 
            block2split->vmm_segment->free_blocks == block2split->vmm_segment->phy_blocks.size()) {
                    continue;
          }
                
                
          bool block_free = block2split->vmm_segment->phy_blocks[0]->free;
          size_t last_offset = 0;
          Block* prev_block = block2split->prev;
                
          auto phy_blocks = block2split->vmm_segment->phy_blocks;
          auto vmm_segment = std::move(block2split->vmm_segment);
                
          for(size_t i=1; i <= phy_blocks.size(); i++) {
                    
            if(i == phy_blocks.size() || block_free != phy_blocks[i]->free) {
              size_t block_size = (i - last_offset)*kGranularity;
                        
              char* block_ptr = (char*)block2split->ptr + last_offset*kGranularity;
              Block* split_block = new Block(p.device(), p.stream(), block_size, p.pool, block_ptr);
                        
                        
              split_block->prev = prev_block;
              if(prev_block) {
                prev_block->next = split_block;
              }
              split_block->self_last_event = block2split->self_last_event;
                        
                        
              if(i < phy_blocks.size()) {
                auto remaining_segment = vmm_segment->split(block_size);
                split_block->vmm_segment = std::move(vmm_segment);
                vmm_segment = std::move(remaining_segment);
              } else {
                split_block->vmm_segment = std::move(vmm_segment);
              }
                        
                        
              size_t offset = 0;
              for(auto& phy_block : split_block->vmm_segment->phy_blocks) {
                phy_block->mapped_blocks[0].block = split_block;
                phy_block->mapped_blocks[0].offset = offset;
                offset++;
              }


              if(block_free) {
                split_block->vmm_segment->free_blocks = split_block->vmm_segment->phy_blocks.size();
                split_block->vmm_segment->used_blocks = 0;
                            
                            
                large_blocks.blocks.insert(split_block);
                            
                            
                net_change_inactive_split_blocks += 1;
                net_change_inactive_split_size += split_block->size;
              } else {
                split_block->vmm_segment->free_blocks = 0;
                split_block->vmm_segment->used_blocks = 0;
                            
                split_block->allocated = true;
                active_blocks.insert(split_block);
                            
                            
                update_stat_array(stats.active, 1, p.stat_types);
                update_stat_array(stats.active_bytes, split_block->size, p.stat_types);
              }
          

              if(i < phy_blocks.size()) {
                block_free = phy_blocks[i]->free;
              }
              last_offset = i;
              prev_block = split_block;
            }
          }
                
                
          if(prev_block) {
            prev_block->next = block2split->next;
          }
                
          if(block2split->next) {
            block2split->next->prev = prev_block;
          }
                
          delete block2split;
        }
            
        p.block->vmm_segment->free_blocks = (p.block->vmm_segment->phy_blocks.size() - keep_blocks);
        p.block->vmm_segment->used_blocks = keep_blocks;

              
        free_fused_blocks.blocks.erase(block_it);
        free_fused_blocks_in_release_order[p.block->stream].erase(p.block);
    
        p.err = cudaSuccess;
    
        update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
        update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
    
        return true;
      }
        
      return false;
    }
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    if (vmmDefragment > 0 && p.block->vmm_segment) {
      for(size_t i=0; i < p.block->vmm_segment->phy_blocks.size(); i++) {
        auto& phy_block = p.block->vmm_segment->phy_blocks[i];
        if(!phy_block->free) {
          GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                      i, p.block, p.block->ptr, p.block->size/(1024.f*1024.f), p.block->vmm_segment->free_blocks, p.block->vmm_segment->used_blocks, p.block->self_last_event->event_id);
            
              
          for(auto& block_segment : phy_block->mapped_blocks) {
            Block* other_block = block_segment.block;
                  
            if(other_block == p.block) continue;
              
            GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, co-ref block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                        i, other_block, other_block->ptr, other_block->size/(1024.f*1024.f), other_block->vmm_segment->free_blocks, other_block->vmm_segment->used_blocks, other_block->self_last_event->event_id);
          }
      
          exit(-1);
        }
          
        phy_block->free = false;
          
        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          if(other_block == p.block) continue;
              
          if(other_block->vmm_segment->fused) {
            if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
              free_fused_blocks.blocks.count(other_block)) {
              free_fused_blocks.blocks.erase(other_block);
              free_fused_blocks_in_release_order[other_block->stream].erase(other_block);

              fragmented_free_fused_blocks[other_block->stream].insert(other_block);
            } else if(active_fused_blocks.count(other_block) == 0) {
              if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              }
            }
                                
                  
            other_block->vmm_segment->free_blocks--;
          } else {
            GMLAKE_INFO(" warning for non fused blocks has phy_block mapped to other non fused blocks");
            exit(-1);
          }
        }
                        
      }
      p.block->vmm_segment->free_blocks = 0;
      p.block->vmm_segment->used_blocks = p.block->vmm_segment->phy_blocks.size();

      
    }
    
    
    
    p.err = cudaSuccess;
    
    
    update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
    update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);
    return true;
#else
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;

    if ((*it)->expandable_segment_) {
      if (CachingAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](Block* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Rarely expandable segments has been turned off after we have
        // already allocated some blocks as expandable. For instance,
        // since we cannot share expandable memory via IPC, someone might
        // temporarily disable it. In this case we need to honor this request
        // by only finding non-expandable blocks
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;

#ifdef DEFRAGMENT
    // outers mem request
    if(!c10::dtb::in_runtime_record[c10::cuda::current_device()]&&UNIFIED_EVICT) {
      // time_t begin = std::chrono::system_clock::now();

      auto first_fit_size = ((*it)->size);
      auto best_it = it;
      size_t best_total_size = std::numeric_limits<size_t>::max();
      float best_frag_ratio = std::numeric_limits<float>::max();

      while(it!=pool.blocks.end() && (*it)->size >= first_fit_size && (*it)->size <= (1.1)*first_fit_size) {
        auto seg = segManager.get_segment_of_block((*it)->ptr);
        seg->flush_frag_ratio();
        float frag_ratio = seg->frag_ratio;
        size_t total_size = seg->total_size; // 假设segment有total_size成员

        if (total_size < best_total_size || (total_size == best_total_size && frag_ratio < best_frag_ratio)) {
          best_total_size = total_size;
          best_frag_ratio = frag_ratio;
          best_it = it;
        }
        it++;
      }

      // time_t end = std::chrono::system_clock::now();
      // auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
      // std::cout << "[sAlloc]" << time_cost << "us" << std::endl;

      if (best_it != pool.blocks.end()) {
        p.block = *best_it;
        (*best_it)->gc_count = 0; // Denote this block has been used
        pool.blocks.erase(best_it);
        return true;
      }
    }
#endif


    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
#endif
  }

  bool get_smaller_free_block(AllocParams& p, size_t max_size) {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }

    auto it = pool.blocks.lower_bound(&p.search_key);     // return first ge search_key container

    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;

    if ((*it)->expandable_segment_) {
      TORCH_INTERNAL_ASSERT(0, "not support yet");
    }

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
#ifdef DEFRAGMENT
    /// FIXME[√]: 这里限制的是被标记的segment，而不应该只是单纯用size做限制
    auto seg = segManager.get_segment_of_block((*it)->ptr);
    while((void*)seg==c10::dtb::move_defrag_seg_ptr[current_device()]) {
      it++;
      if(it==pool.blocks.end()) {
        return false;
      }
      seg = segManager.get_segment_of_block((*it)->ptr);
    }
#endif

    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;

  }

#ifdef GMLAKE_ENABLE
  size_t garbage_collect_fused_blocks(int time, size_t require_size = 0) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
      
    size_t garbage_size = 0;
    size_t garbage_blocks = 0;
    for(auto& it : fragmented_free_fused_blocks) {
      for(auto block_it = it.second.blocks.begin(); block_it != it.second.blocks.end();) {
        Block* block = (*block_it);
      
        cudaError_t err = cudaSuccess;
        if(block->self_last_event) {
          err = cudaEventQuery(block->self_last_event->event);
        }
              
        if(err == cudaSuccess) {
          for(auto& phy_block : block->vmm_segment->phy_blocks) {
            int i = 0;
            for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
              if(phy_block->mapped_blocks[j].block != block) {
                if(i != j) {
                  phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                }
                              
                i++;
              }
            }
            phy_block->mapped_blocks.resize(i);
          }
                  
          garbage_blocks++;
          garbage_size += block->size;
                  
          //free_fused_blocks.blocks.erase(block);
          block_it = it.second.erase(block_it);
                  
                  
          if(!block->vmm_segment.unique()) {
            GMLAKE_INFO(" warning block is not unique, ref_count: %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block->vmm_segment.use_count(), block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
            exit(-1);
          }
                  
                  
          if(block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count() != block->vmm_segment->vir_blocks.size()) {
            GMLAKE_INFO(" warning vir_blocks vir_dev_ptr use_count %lu != vir_blocks.size() %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count(), block->vmm_segment->vir_blocks.size(),
                        block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
            exit(-1);
          }
                  
          {
            //block->vmm_segment.reset();
            auto tmp = std::move(block->vmm_segment);
          }
                  
          delete block;
                  
          if(require_size > 0 && time <= 1 && garbage_size >= (require_size << (2*(time + 1))) ) break;
          
        } else if(err == cudaErrorNotReady) {
                  
          GMLAKE_INFO(" fragmented_free_fused_blocks: block self_last_event NotReady %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                      block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
                  
                  
          cudaGetLastError();
          break;
        } else {
          C10_CUDA_CHECK(err);
          cudaGetLastError();
          break;
        }
      }
    }
      
      
    GMLAKE_INFO(" gc from fragmented_free_fused_blocks: blocks %lu, size %fMB", garbage_blocks, garbage_size/(1024.f*1024.f));
      
      
    if(time > 0) {
      for(auto& it : free_fused_blocks_in_release_order) {
        for(auto block_it = it.second.blocks.begin(); block_it != it.second.blocks.end();) {
          Block* block = (*block_it);
      
          cudaError_t err = cudaSuccess;
          if(block->self_last_event) {
            err = cudaEventQuery(block->self_last_event->event);
          }
                
          if(err == cudaSuccess) {
            for(auto& phy_block : block->vmm_segment->phy_blocks) {
              int i = 0;
              for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
                if(phy_block->mapped_blocks[j].block != block) {
                  if(i != j) {
                    phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                  }
                                
                  i++;
                }
              }
                        
              phy_block->mapped_blocks.resize(i);
            }
                    
            garbage_blocks++;
            garbage_size += block->size;
                    
            free_fused_blocks.blocks.erase(block);
            block_it = it.second.erase(block_it);
                  
                            
            if(!block->vmm_segment.unique()) {
              GMLAKE_INFO(" warning block is not unique, ref_count %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                          block->vmm_segment.use_count(), block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
              exit(-1);
            }
                    
                    
                  
            if(block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count() != block->vmm_segment->vir_blocks.size()) {
              GMLAKE_INFO(" warning vir_blocks vir_dev_ptr use_count %lu != vir_blocks.size() %lu, block %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                          block->vmm_segment->vir_blocks[0]->vir_dev_ptr.use_count(), block->vmm_segment->vir_blocks.size(),
                          block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
              exit(-1);
            }
                    
            delete block;
          } else if(err == cudaErrorNotReady) {
            GMLAKE_INFO(" free_fused_blocks_in_release_order: block self_last_event NotReady %p, block->ptr %p, block->size %fMB, phy_blocks %lu, free_blocks %lu, used_blocks %lu, event_id: %lu", 
                        block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->phy_blocks.size(), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
                    
            cudaGetLastError();
            break;
          } else {
            C10_CUDA_CHECK(err);
            cudaGetLastError();
            break;
          }
        }
      }   
    }
      
      //cudaDeviceSynchronize();
      
    GMLAKE_INFO(" gc from free_fused_blocks_in_release_order: blocks %lu, size %fMB", garbage_blocks, garbage_size/(1024.f*1024.f));

    return garbage_size;
  }

  bool get_fused_fragmented_blocks(AllocParams& p, int time) {
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();
    
    
    static const size_t fragment_limit = ([]()->size_t{
        const char* env = getenv("fragLimit");
        if(env) return (size_t)std::stoll(env);
        else return (size_t)(512*1024*1024);
    })();
    
    
    static const int defragment_level = ([]()->int{
        const char* env = getenv("defragLevel");
        if(env) return (int)std::atoi(env);
        else return (int)0;
    })();
    
    
    static const int auto_gc_limits = ([]()->int{
        const char* env = getenv("autoGC");
        if(env) return (int)std::atoi(env);
        else return (int)1000;
    })();
    
    
    if (vmmDefragment <= 0) {
      return false;
    }
    
    
    if(time < defragment_level) {
        return false;
    }
    
    
    if (p.pool->is_small || p.search_key.size < fragment_limit) {
      return false;
    } else {
      Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
      Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);

      left_search_key.size = 0;
      right_search_key.size = std::numeric_limits<size_t>::max();

      auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
      if (it_begin == large_blocks.blocks.end() || (*it_begin)->stream != p.stream())
        return false;
      
      auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
      if (it_end == large_blocks.blocks.begin() || (*std::prev(it_end))->stream != p.stream())
        return false;
      
      
      if(std::prev(it_end) == it_begin) return false;
      
      size_t fuse_size = 0;
      std::vector<Block*> blocks2fuse;
      
      auto it = it_end;
      while(it != it_begin && fuse_size < p.search_key.size) {
        it = std::prev(it);
        blocks2fuse.push_back((*it));
        fuse_size += (*it)->size;
      }
      
      
      if(fuse_size < p.search_key.size) {
          return false;
      }
            
      
      int64_t net_change_segments = 0;
      int64_t net_change_inactive_split_blocks = 0;
      int64_t net_change_inactive_split_size = 0;
      
      
      std::shared_ptr<BlockEvent> current_self_last_event;
      std::vector<std::shared_ptr<PhyBlock>> phy_blocks2glue;
      int index = 0;  
      for(auto& block : blocks2fuse) {
        for(auto& phy_block : block->vmm_segment->phy_blocks) {
          phy_block->free = false;
          phy_blocks2glue.push_back(phy_block);
        }
        block->vmm_segment->free_blocks = 0;
        block->vmm_segment->used_blocks = 0;

        
        if(!current_self_last_event || 
          (block->self_last_event && block->self_last_event->event_id > current_self_last_event->event_id)) {
          current_self_last_event = block->self_last_event;
        }
        
        large_blocks.blocks.erase(block);
        
        
        if(block->is_split()) {
          net_change_inactive_split_blocks -= 1;
          net_change_inactive_split_size -= block->size;
        }
        
        
        
        block->allocated = true;
        active_blocks.insert(block);
        
        if (index == blocks2fuse.size() - 1 && (fuse_size - p.search_key.size) >= kGranularity) continue;
        update_stat_array(stats.active, 1, p.stat_types);
        update_stat_array(stats.active_bytes, block->size, p.stat_types);
        index++;
      }

      if(fuse_size > p.search_key.size && (fuse_size - p.search_key.size) >= kGranularity) {
        Block* last_block = blocks2fuse.back();
          
          
        last_block->allocated = false;
        if(active_blocks.count(last_block)) {
          active_blocks.erase(last_block);
        }
                      
          
        Block* remaining = last_block;
      
        size_t original_size = remaining->size;
        size_t remain_size = (fuse_size - p.search_key.size);
        size_t keep_size = original_size - remain_size;
  
        last_block = new Block(p.device(), p.stream(), keep_size, p.pool, last_block->ptr);
        last_block->prev = remaining->prev;
        if (last_block->prev) {
            last_block->prev->next = last_block;
        }
        last_block->next = remaining;
        last_block->self_last_event = remaining->self_last_event;
          
        remaining->prev = last_block;
        remaining->ptr = static_cast<char*>(remaining->ptr) + keep_size;
        remaining->size = remain_size;
          
        auto remaining_segment = remaining->vmm_segment->split(keep_size);
        last_block->vmm_segment = std::move(remaining->vmm_segment);
        remaining->vmm_segment =  std::move(remaining_segment);
          
        for(size_t i=0; i<last_block->vmm_segment->phy_blocks.size(); i++) {
          last_block->vmm_segment->phy_blocks[i]->mapped_blocks[0].block = last_block;
          last_block->vmm_segment->phy_blocks[i]->mapped_blocks[0].offset = i;
          last_block->vmm_segment->phy_blocks[i]->free = false;
        }
        last_block->vmm_segment->free_blocks = 0;
        last_block->vmm_segment->used_blocks = 0;
        last_block->allocated = true;
                    
        active_blocks.insert(last_block);

        update_stat_array(stats.active, 1, p.stat_types);
        update_stat_array(stats.active_bytes, last_block->size, p.stat_types);
          
        for(size_t i=0; i<remaining->vmm_segment->phy_blocks.size(); i++) {
          remaining->vmm_segment->phy_blocks[i]->mapped_blocks[0].block = remaining;
          remaining->vmm_segment->phy_blocks[i]->mapped_blocks[0].offset = i;
          remaining->vmm_segment->phy_blocks[i]->free = true;
        }
        remaining->vmm_segment->free_blocks = remaining->vmm_segment->phy_blocks.size();
        remaining->vmm_segment->used_blocks = 0;
        remaining->allocated = false;
          
        large_blocks.blocks.insert(remaining);
          
        fuse_size -= remaining->size;
  
        size_t keep_blocks = p.search_key.size/kGranularity;
        phy_blocks2glue.resize(keep_blocks);
            
        net_change_inactive_split_blocks += 1;
        net_change_inactive_split_size += remaining->size;
      }
      
      static constexpr size_t G=1024*1024*1024;
      
      using Ms = std::chrono::duration<double, std::milli>;
      Ms fuse_time = Ms{0};
      
      std::shared_ptr<VmmSegment> vmm_segment;
      int gc_time = 0;
      do
      {
        auto t0 = std::chrono::steady_clock::now();
          
        vmm_segment = std::make_shared<VmmSegment>(std::move(phy_blocks2glue));
          
        auto t1 = std::chrono::steady_clock::now();
        fuse_time = (t1-t0);
          
        if(vmm_segment->status == CUDA_SUCCESS && vmm_segment->segment_ptr) {
          break;
        } else {
          cudaGetLastError();
              
          phy_blocks2glue = std::move(vmm_segment->phy_blocks);
              
          GMLAKE_INFO(" allocate virtual address for %lu phy_blocks the %dth time failed, try to garbage_collect_fused_blocks", phy_blocks2glue.size(), gc_time);
              
          size_t garbage_size = garbage_collect_fused_blocks(gc_time, p.search_key.size);
          gc_time++;
              
          total_fuse_size -= garbage_size;
              
          cudaGetLastError();
        }
      } while(gc_time < 3);
      
      if(!vmm_segment || vmm_segment->status != CUDA_SUCCESS || !vmm_segment->segment_ptr) {
          return false;
      }
      
      void* block_ptr = vmm_segment->segment_ptr;
      Block* fused_block = new Block(p.device(), p.stream(), fuse_size, p.pool, (char*)block_ptr);

      fused_block->vmm_segment = std::move(vmm_segment);
      fused_block->self_last_event = current_self_last_event;

      for(auto& phy_block : fused_block->vmm_segment->phy_blocks) {
        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          //since the non fused blocks has already been processed, we only need to process fused blocks 
          if(other_block->vmm_segment->fused) {
            if(other_block->vmm_segment->free_blocks == other_block->vmm_segment->phy_blocks.size() && 
              free_fused_blocks.blocks.count(other_block)) {
              free_fused_blocks.blocks.erase(other_block);
              free_fused_blocks_in_release_order[other_block->stream].erase(other_block);
        
              fragmented_free_fused_blocks[other_block->stream].insert(other_block);
            } else if(active_fused_blocks.count(other_block) == 0) {
              if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block) == 0) {
                fragmented_free_fused_blocks[other_block->stream].insert(other_block);
              }
            }
                  

            other_block->vmm_segment->free_blocks--;
          }
        }
      }

      size_t offset = 0;
      for(auto& phy_block : fused_block->vmm_segment->phy_blocks) {
        phy_block->mapped_blocks.emplace_back(fused_block, offset);
        offset++;
      }
      fused_block->vmm_segment->free_blocks = 0;
      fused_block->vmm_segment->used_blocks = fused_block->vmm_segment->phy_blocks.size();

      p.block = fused_block;
      p.err = cudaSuccess;

      GMLAKE_INFO(" fused block %p, ptr %p of size %fMB", 
			            fused_block, fused_block->ptr, fused_block->size/(1024.f*1024.f));
      
      net_change_segments += 1;


      update_stat_array(stats.segment, net_change_segments, p.stat_types);
      update_stat_array(stats.inactive_split, net_change_inactive_split_blocks, p.stat_types);
      update_stat_array(stats.inactive_split_bytes, net_change_inactive_split_size, p.stat_types);

      if(fuse_size >= p.search_key.size) {
        total_fuse_size += fuse_size;
        GMLAKE_INFO(" try %d: fuse %lu physical blocks to ptr %p of size %fMB for allocate size %fMB succeeded, takes %fms, total_fuse_size %fMB", 
                   time, fused_block->vmm_segment->phy_blocks.size(), fused_block->vmm_segment->segment_ptr, fuse_size/(1024.f*1024.f), p.search_key.size/(1024.f*1024.f), fuse_time.count(), total_fuse_size/(1024.f*1024.f));
        
        if(total_fuse_size > auto_gc_limits*G) {
            GMLAKE_INFO(" virtual address larger than %luG, do garbage_collect_fused_blocks() ", auto_gc_limits);
            
            size_t garbage_size = garbage_collect_fused_blocks(2, 0);
            
            total_fuse_size -= garbage_size;
        }
      }

      return fuse_size >= p.search_key.size;
    } 
    
    return false;
  }
  
#endif

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count; // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block);
        }
      }
    }
  }

#ifdef GMLAKE_ENABLE
  bool realloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();


    static const int reAlloc = ([]()->int{
        const char* env = getenv("reAlloc");
        if(env) return atoi(env);
        else return 0;
    })();



    size_t size = p.alloc_size;
    size_t free_block_size = 0;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    std::shared_ptr<VmmSegment> vmm_segment;
    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      if(vmmDefragment <= 0 || p.pool->is_small) {
        p.err = cudaMallocMaybeCapturing(&ptr, size);
        if (p.err != cudaSuccess) {
          if (p.err == cudaErrorMemoryAllocation) {
            // If this is the first attempt (!isRetry), we can forgive and clear CUDA's
            //   internal error state.
            // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH will take
            //   over to throw a helpful exception. The user can choose to catch the exception,
            //   free some stuff in their script, and attempt their allocation again.
            //   In this case, we can also forgive and clear CUDA's internal error state.
            cudaGetLastError();
          } else {
            // If the error's unrelated to memory allocation, we should throw immediately.
            C10_CUDA_CHECK(p.err);
          }
          return false;
        }
      } else {
        if(reAlloc > 0) {
          //Block left_search_key = p.search_key;
          //Block right_search_key = p.search_key;
                
          Block left_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
          Block right_search_key(p.search_key.device, p.search_key.stream, p.search_key.size, p.search_key.pool, p.search_key.ptr);
      
          left_search_key.size = 0;
          right_search_key.size = std::numeric_limits<size_t>::max();
                      
          auto it_begin = large_blocks.blocks.lower_bound(&left_search_key);
          auto it_end = large_blocks.blocks.lower_bound(&right_search_key);
                
          if(it_begin != large_blocks.blocks.end() && (*it_begin)->stream == p.stream() &&
            it_end != large_blocks.blocks.begin() && (*std::prev(it_end))->stream == p.stream()) {
            auto it = it_begin;
            while(it != it_end) {
              free_block_size += (*it)->size;
              it++;
            }
          }
                
                
          size_t request_size = p.search_key.size;
                
          if(free_block_size >= request_size) {
            GMLAKE_INFO(" free_block_size %fMB is larger than allocation size %fMB, something weired happended", 
                       free_block_size/(1024.f*1024.f), size/(1024.f*1024.f));
            return false;
          }
                
          if(free_block_size > 0) {
            request_size -= free_block_size;
            size = get_allocation_size(request_size);
          }
        }
               
        using Ms = std::chrono::duration<double, std::milli>;
        Ms fuse_time = Ms{0};
            
        int gc_time = 0;
        do
        {
          auto t0 = std::chrono::steady_clock::now();
                
          vmm_segment = std::make_shared<VmmSegment>(size/kGranularity, kGranularity, p.device());
                
          auto t1 = std::chrono::steady_clock::now();
          fuse_time = (t1-t0);
                
          if(vmm_segment->status == CUDA_SUCCESS && vmm_segment->segment_ptr) {
            break;
          } else {
            cudaGetLastError();
                            
            size_t device_free;
            size_t device_total;
            cudaMemGetInfo(&device_free, &device_total);
                            
            size_t total_garbage_size = fragmented_free_fused_blocks[p.stream()].pool_size + free_fused_blocks_in_release_order[p.stream()].pool_size;
                  
                    
            if(device_free > size && total_garbage_size >= size) {
              GMLAKE_INFO(" allocate size %luMB memory by vmm the %dth time failed, try to garbage_collect_fused_blocks", size/(1024*1024), gc_time);
                        
              vmm_segment.reset();
              size_t garbage_size = garbage_collect_fused_blocks(gc_time, p.alloc_size);
              total_fuse_size -= garbage_size;
                        
              gc_time++;
                       
              cudaGetLastError();
            } else {
              break;
            }
          }
        } while(gc_time < 3);
            
        if(!vmm_segment || vmm_segment->status != CUDA_SUCCESS || !vmm_segment->segment_ptr) {           
          p.err = cudaErrorMemoryAllocation;
          cudaGetLastError();
          vmm_segment.reset();
                
          GMLAKE_INFO(" allocate size %fMB memory by vmm failed", size/(1024.f*1024.f));
            
          return false;
        }
            
            
        ptr = vmm_segment->segment_ptr;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    Block* new_block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    new_block->vmm_segment = std::move(vmm_segment);
    
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    //TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    TORCH_INTERNAL_ASSERT(new_block != nullptr && new_block->ptr != nullptr);
    
    
    if(new_block->vmm_segment) {
      if(new_block->size < p.search_key.size) {
        for(size_t i = 0; i < new_block->vmm_segment->phy_blocks.size(); i++) {
          new_block->vmm_segment->phy_blocks[i]->mapped_blocks.emplace_back(new_block, i);
          new_block->vmm_segment->phy_blocks[i]->free = true;
        }
            
        new_block->vmm_segment->free_blocks = new_block->vmm_segment->phy_blocks.size();
        new_block->vmm_segment->used_blocks = 0;

        large_blocks.blocks.insert(new_block);
            
        if(!get_fused_fragmented_blocks(p, 4)) {
          GMLAKE_INFO(" call get_fused_fragmented_blocks failed");
          return false;
        }
      } else {
        for(size_t i = 0; i < new_block->vmm_segment->phy_blocks.size(); i++) {
          new_block->vmm_segment->phy_blocks[i]->mapped_blocks.emplace_back(new_block, i);
          new_block->vmm_segment->phy_blocks[i]->free = false;
        }
            
        new_block->vmm_segment->free_blocks = 0;
        new_block->vmm_segment->used_blocks = new_block->vmm_segment->phy_blocks.size();
            
        p.block = new_block;
        p.err = cudaSuccess;
      }
    } else {
      p.block = new_block;
      p.err = cudaSuccess;
    }
    
    return true;
  }
#endif

  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<GatheredContext>& ctx) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else if (
        CachingAllocatorConfig::expandable_segments() &&
        // our checkpointing logic for private pools doesn't support
        // the expandable_segments_ structure yet
        !p.pool->owner_PrivatePool) {
      p.block = try_allocate_expandable_block(
          p.device(), p.stream(), p.pool, p.size(), ctx);
      if (p.block) {
        p.err = cudaSuccess;
      } else {
        p.err = cudaErrorMemoryAllocation;
      }
      return bool(p.block);
    } else {
      p.err = cudaMallocMaybeCapturing(&ptr, size);
      #ifdef MEM_EVENTS_REC
      if(log_cudaAPI) DTRLogcudaAPIEvents("cudaMalloc", size, reinterpret_cast<uintptr_t>(ptr));
      #endif
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's internal error state.
          //
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take over to throw a helpful exception. The user can choose
          // to catch the exception, free some stuff in their script, and
          // attempt the allocation again. In this case, we can also forgive and
          // clear CUDA's internal error state.
          (void)cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
#ifdef MEM_TWIN_REC
    if(UNIFIED_EVICT){
      SegmentTwin* new_seg = new SegmentTwin(p.block);    // [TAG] only entry for new SegmentTwin creatation
      segManager.insert(new_seg);
      segManager.add_block2segment(p.block, new_seg);
    #ifdef MEM_TWIN_DEBUG
      printf("[CREATE] seg_members:%ld, erase:%ld\n", new_seg->blocks.size(), reinterpret_cast<uintptr_t>(p.block->ptr));
    #endif
    }
#endif
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    if(log_mem_events) {
      record_mem_events(
          TraceEntry::SEGMENT_ALLOC,
          int64_t(p.block->ptr),
          p.block->size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::SEGMENT_ALLOC,
          int64_t(p.block->ptr),
          p.block->size,
          p.stream(),
          ctx);
      p.block->context_when_segment_allocated = ctx;
    }
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    Block key(
        p.search_key.device,
        p.search_key.stream,
        p.search_key.size,
        p.search_key.pool,
        p.search_key.ptr);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(context);

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);
#ifdef MORE_POOL
    release_blocks(ex1_blocks);
    release_blocks(ex2_blocks);
#endif

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      // See notifyCaptureDestroy for the strategy here.
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks);
      release_blocks(it->second->large_blocks);
      if (it->second->cudaMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  void release_expandable_segment(Block* block) {
    TORCH_INTERNAL_ASSERT(
        block->size == block->expandable_segment_->size(),
        "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);
    auto it = std::find(
        expandable_segments_.begin(),
        expandable_segments_.end(),
        block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    delete block;
  }

  void release_block(Block* block) {
#ifdef GMLAKE_ENABLE
static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();



    if (vmmDefragment > 0 && block->vmm_segment) {
      for(size_t i=0; i < block->vmm_segment->phy_blocks.size(); i++) {
        auto& phy_block = block->vmm_segment->phy_blocks[i];
        if(!phy_block->free) {
          GMLAKE_INFO(" warning for non fused blocks has non free phy_block: %lu, something wrong happended, block %p, block->ptr %p, block->size %fMB, free_blocks %lu, used_blocks %lu, event_id: %lu",
                                               i, block, block->ptr, block->size/(1024.f*1024.f), block->vmm_segment->free_blocks, block->vmm_segment->used_blocks, block->self_last_event->event_id);
        }

        for(auto& block_segment : phy_block->mapped_blocks) {
          Block* other_block = block_segment.block;
              
          if(other_block == block) continue;
              
          if(other_block->vmm_segment->fused) {
            if(active_fused_blocks.count(other_block) && 
              active_fused_blocks_to_gc.count(other_block) == 0) {
              {
                auto tmp1 = std::move(other_block->vmm_segment->vir_blocks[block_segment.offset]);
                auto tmp2 = std::move(other_block->vmm_segment->phy_blocks[block_segment.offset]);
              }
                      
              //active_fused_blocks.erase(other_block);
              active_fused_blocks_to_gc.insert(other_block);
            } else if(free_fused_blocks.blocks.count(other_block) || 
                          fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
              if(free_fused_blocks.blocks.count(other_block)) {
                free_fused_blocks.blocks.erase(other_block);
                free_fused_blocks_in_release_order[other_block->stream].erase(other_block);
              } else if(fragmented_free_fused_blocks[other_block->stream].blocks.count(other_block)) {
                fragmented_free_fused_blocks[other_block->stream].erase(other_block);
              }
       
              for(auto& phy_block : other_block->vmm_segment->phy_blocks) {
                int i = 0;
                for(int j = 0; j < phy_block->mapped_blocks.size(); j++) {
                  if(phy_block->mapped_blocks[j].block != other_block) {
                    if(i != j) {
                      phy_block->mapped_blocks[i] = phy_block->mapped_blocks[j];
                    }
                                  
                    i++;
                  }
                }
                phy_block->mapped_blocks.resize(i);
              }
                      
                      
              delete other_block;
            }
          } else {
            GMLAKE_INFO(" warning for non fused blocks has phy_block mapped to other non fused blocks");
            exit(-1);
          }
        }
                        
      }
      
      
    }
    
    

    
    if(block->vmm_segment){
      block->vmm_segment.reset();
    } else {
      C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    }
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(
          stats.reserved_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);
    if (block->history) {
      record_trace(
          TraceEntry::SEGMENT_FREE,
          int64_t(block->ptr),
          block->size,
          block->stream,
          block->history->h.context);
    }
    pool->blocks.erase(block);
    delete block;
#else
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
#ifdef MEM_EVENTS_REC
    if(log_cudaAPI) DTRLogcudaAPIEvents("cudaFree", block->size, reinterpret_cast<uintptr_t>(block->ptr));
#endif
#ifdef MEM_TWIN_REC
    if(UNIFIED_EVICT){
      auto *seg = segManager.get_segment_of_block(block->ptr, true /*remove*/);   // remove rec in block2seg
    #ifdef MEM_TWIN_DEBUG
      printf("[RELEASE] seg_members:%ld, erase:%ld\n", seg->blocks.size(), reinterpret_cast<uintptr_t>(block->ptr));
    #endif
      seg->erase(block);
      TORCH_INTERNAL_ASSERT(seg->empty());    // a block can be released only when it's no split.
      segManager.erase(seg);                  // remove rec in size_map
      delete seg;                             // [TAG] only entry of segment destroying
    }
#endif
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(
          stats.reserved_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);
    if(log_mem_events) {
      record_mem_events(
          TraceEntry::SEGMENT_FREE,
          int64_t(block->ptr),
          block->size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::SEGMENT_FREE,
          int64_t(block->ptr),
          block->size,
          block->stream,
          nullptr);
    }
    pool->blocks.erase(block);
    delete block;
#endif
  }

  void unmap_block(Block* block) {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size =
        static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      Block* before_free = new Block(
          block->device, block->stream, before_size, block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->blocks.insert(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block* after_free = new Block(
          block->device,
          block->stream,
          after_size,
          block->pool,
          static_cast<char*>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->blocks.insert(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], -unmapped.size);
    });
    if(log_mem_events) {
      record_mem_events(TraceEntry::SEGMENT_UNMAP,
          int64_t(unmapped.ptr),
          unmapped.size);
    }
    if (record_history) {
      record_trace(
          TraceEntry::SEGMENT_UNMAP,
          int64_t(unmapped.ptr),
          unmapped.size,
          block->stream,
          nullptr);
    }
  }
  
  void release_blocks(BlockPool& pool) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  EventPool::Event create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context) {
    // Synchronize on outstanding events and then free associated blocks.

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway == 0);
    insert_events_deferred_until_no_capture();

    for (auto& st : cuda_events) {
      for (auto& e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        C10_CUDA_CHECK(cudaEventSynchronize(*event));

        block->event_count--;
        if (block->event_count == 0) {
#ifdef GMLAKE_ENABLE
          update_block(block, context);
#else
          free_block(block, context);
#endif
        }
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block) {
    int prev_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));

      EventPool::Event event =
          create_event_internal(static_cast<int>(stream.device_index()));
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      block->event_count++;
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
  }

#ifdef GMLAKE_ENABLE
  void insert_free_event_into_alloc_stream(Block* block)
  {
    int prev_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));
    if(prev_device != block->device)
    {
      C10_CUDA_CHECK(cudaSetDevice(block->device));
    }
  

    
    if( block->self_last_event && 
        block->self_last_event.unique() && 
        block->self_last_event->stream == block->stream && 
        !block->self_last_event->ref_as_sync) {
      block->self_last_event->record(block->stream);
    } else {
      block->self_last_event = std::make_shared<BlockEvent>(block->stream, true);
    }

    if(prev_device != block->device) {
      C10_CUDA_CHECK(cudaSetDevice(prev_device));
    }
  }
#endif

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events(const std::shared_ptr<GatheredContext>& context) {
    insert_events_deferred_until_no_capture();

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // ignore and clear the error if not ready
          (void)cudaGetLastError();
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
        }

        block->event_count--;
        if (block->event_count == 0) {
#ifdef GMLAKE_ENABLE
          update_block(block, context);
#else
          free_block(block, context);
#endif
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void record_mem_events(TraceEntry::Action action, int64_t addr, size_t size) {
    if(last_flag!=c10::dtb::during_backward){
      if(!last_flag&&c10::dtb::during_backward) // 开始反向
        DTRLogger::logger().log("{\"TYPE\":\"begin backward\"}");
      else
        DTRLogger::logger().log("{\"TYPE\":\"end backward\"}");
      last_flag = c10::dtb::during_backward;
    }
    DTRLogMemEvents(std::to_string(action), size, addr);
  }

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      cudaStream_t stream,
      std::shared_ptr<GatheredContext> context) {
    
// #ifdef MEM_EVENTS_REC
//     if(log_mem_events){
//       if(last_flag!=c10::dtb::during_backward){
//         if(!last_flag&&c10::dtb::during_backward) // 开始反向
//           DTRLogger::logger().log("{\"TYPE\":\"begin backward\"}");
//         else
//           DTRLogger::logger().log("{\"TYPE\":\"end backward\"}");
//         last_flag = c10::dtb::during_backward;
//       }
//       DTRLogMemEvents(std::to_string(action), size, addr);
//     }
// #endif
    auto te = TraceEntry(
        action,
        addr,
        size,
        stream,
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(te);
    } else {
      (*alloc_trace)[alloc_trace_next++] = te;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }
};

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cudaMalloc.  This setting is useful when debugging GPU memory
// errors, since the caching allocator foils cuda-memcheck.
bool forceUncachedAllocator() {
  static bool force_uncached =
      getenv("PYTORCH_NO_CUDA_MEMORY_CACHING") != nullptr;
  return force_uncached;
}

static void uncached_delete(void* ptr) {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_memory_deallocation(reinterpret_cast<uintptr_t>(ptr));
  }
  C10_CUDA_CHECK(cudaFree(ptr));
}

void local_raw_delete(void* ptr);

class NativeCachingAllocator : public CUDAAllocator {
 private:
  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
#ifdef GMLAKE_ENABLE
    static const int vmmDefragment = ([]()->int{
        const char* env = getenv("vmmDefragment");
        if(env) return atoi(env);
        else return 1;
    })();
    if (vmmDefragment) {
        GMLAKE_INFO(" GMLAKE initialized");
    }
#endif
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  bool initialized() override {
    return !device_allocator.empty();
  }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, cudaStream_t stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = (void*)block->ptr;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_allocation(
          reinterpret_cast<uintptr_t>(*devPtr));
    }
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_memory_deallocation(
          reinterpret_cast<uintptr_t>(block->ptr));
    }
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    C10_CUDA_CHECK(c10::cuda::SetDevice(device));
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    for (auto& allocator : device_allocator) {
      allocator->recordHistory(
          enabled, context_recorder, alloc_trace_max_entries, when);
    }
  }

  bool isHistoryEnabled() override {
    int device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    return device_allocator[device]->isHistoryEnabled();
  }

  bool checkPoolLiveAllocations(
      int device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return device_allocator[device]->checkPoolLiveAllocations(
        mempool_id, expected_live_allocations);
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    for (auto& allocator : device_allocator) {
      allocator->attachOutOfMemoryObserver(std::move(observer));
    }
  }

  void emptyCache() override {
    for (auto& da : device_allocator)
      da->emptyCache();
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const DataPtr& ptr, cuda::CUDAStream stream) override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete)
      return;

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }

  SnapshotInfo snapshot() override {
    SnapshotInfo result;
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace());
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }
    return result;
  }

  std::shared_ptr<AllocatorState> getCheckpointState(int device, MempoolId_t id)
      override {
    return device_allocator[device]->getCheckpointState(id);
  }

#ifdef MEM_TWIN_REC
  // TODO: provide some interface for outer access like how `getCheckpointState` do
  // void getSegmentsSnapshot(int device) {
  //   device_allocator[device]->getSegmentTwins();
  // }

#endif

  /**
   * @brief Checkpoint the private pool state identified in `as` to its prior
   * state
   *
   * @param device - device of the pool to manipulate
   * @param as - allocator state
   * @param stale_live_storages - storages of tensors which are currently
   * allocated but which will be not be allocated after the checkpoint is set.
   * For these storages we will remove their deleter function.
   * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
   * functions for all allocated blocks in the new checkpoint state.
   */
  CheckpointDelta setCheckpointPoolState(
      int device,
      std::shared_ptr<AllocatorState> as) override {
    std::shared_ptr<PrivatePoolState> pps =
        std::dynamic_pointer_cast<PrivatePoolState>(as);

    TORCH_CHECK(pps, "Expected PrivatePoolState");

    auto rr = device_allocator[device]->setCheckpointPoolState(*pps);

    CheckpointDelta cpd;
    for (void* ptr : rr.allocations_freed) {
      get_allocated_block(ptr, /*remove*/ true);
      cpd.ptrs_freed.push_back(ptr);
    }
    for (Block* block : rr.allocations_created) {
      add_allocated_block(block);
      cpd.dataptrs_allocd.emplace_back(
          block->ptr,
          block->ptr,
          &local_raw_delete,
          Device(DeviceType::CUDA, device));
    }

    return cpd;
  }

  DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    int device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* r = nullptr;
    if (forceUncachedAllocator()) {
      // Deliberately don't use cudaMallocMaybeCapturing here, to force an error
      // if someone tries to use forceUncachedAllocator while capturing.
      C10_CUDA_CHECK(cudaMalloc(&r, size));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_memory_allocation(reinterpret_cast<uintptr_t>(r));
      }
      return {r, r, &uncached_delete, Device(DeviceType::CUDA, device)};
    }
    if (size != 0) {
      if (TORCH_SDT_IS_ENABLED(malloc)) {
        TORCH_SDT_WITH_SEMAPHORE(malloc, &r, device, size, 0);
      }

      // Allocator declars allocate const!?
      const_cast<NativeCachingAllocator*>(this)->malloc(
          &r, device, size, cuda::getCurrentCUDAStream(device));
    }
    return {r, r, &local_raw_delete, Device(DeviceType::CUDA, device)};
  }
  DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }
  void cacheInfo(int dev_id, size_t* largestBlock) override {
    device_allocator[dev_id]->cacheInfo(largestBlock);
  }
  void assertValidDevice(int device) {
    const auto device_num = device_allocator.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  DeviceStats getDeviceStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->getStats();
  }

  void logPtrInfo(int device, std::vector<void *> ptrs) override {
    assertValidDevice(device);
    return device_allocator[device]->logPtrInfo(ptrs);
  }

  void resetAccumulatedStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
  }
  // CUDAGraph interactions
  void beginAllocateStreamToPool(
      int device,
      cudaStream_t stream,
      MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->beginAllocateStreamToPool(
        stream, std::move(mempool_id));
  }

  void endAllocateStreamToPool(int device, cudaStream_t stream) override {
    assertValidDevice(device);
    device_allocator[device]->endAllocateStreamToPool(stream);
  }

  void releasePool(int device, MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->releasePool(std::move(mempool_id));
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, cuda::getCurrentCUDAStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }

  void enablePeerAccess(int dev, int dev_to_access) override {
    c10::cuda::CUDAGuard device_guard(dev);
    cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
      // ignore and clear the error if access was already enabled
      (void)cudaGetLastError();
    } else {
      C10_CUDA_CHECK(err);
    }
    device_allocator[dev_to_access]->addPeerAccess(dev);
  }

  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override {
    if (p2p_enabled || // memcpy ok because memory is mapped in both devices
        srcDevice == dstDevice || // memcpy ok on a single device
        // memcpy ok because both dst and src must have come from cudaMalloc
        (!device_allocator[dstDevice]->hasAllocatedExpandableSegments() &&
         !device_allocator[srcDevice]->hasAllocatedExpandableSegments())) {
      return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }
    // when p2p is not enabled, only cudaMemcpyPeerAsync correctly handles
    // memory not allocated via cudaMalloc
    return cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
  }

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  // In CUDA IPC, sender sends a tensor to receiver, getIpcDevPtr
  // is called by the receiving process to map the CUDA memory from the sending
  // process into its own address space.
  //
  // CUDA IPC only allows sharing a big memory block associated with a
  // cudaIpcMemHandle_t and it can be opened only **once** per context per
  // process. There can be multiple types of storage in the same IPC mem block,
  // so we must cache the device ptr to construct typed storage as it comes.
  //
  // ipcMemHandle_to_devptr maps a cudaIpcMemHandle_t to a device pointer in the
  // process that can be used to access the memory block in the sender process.
  // It only saves a weak_ptr of the device pointer in the map, the shared_ptr
  // will be used to reconstruct all storages in this CudaMalloc allocation. And
  // it will deleted in cudaIpcCloseMemHandle when its reference count is 0.
  //
  std::mutex IpcMutex;
  ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex);

    auto iter = ipcMemHandle_to_devptr.find(handle);
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.lock();
      if (devptr)
        return devptr;
    }
    // This ipcMemHandle hasn't been opened, or already expired, open it to
    // enable IPC access to that mem block.
    void* dev = nullptr;
    auto ipc_handle =
        reinterpret_cast<const cudaIpcMemHandle_t*>(handle.c_str());
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &dev, *ipc_handle, cudaIpcMemLazyEnablePeerAccess));
    // devPtr has to be deleted in same device when created.
    int curr_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&curr_device));
    auto sp =
        std::shared_ptr<void>(dev, [handle, curr_device, this](void* ptr) {
          cuda::CUDAGuard device_guard(curr_device);
          std::lock_guard<std::mutex> deleter_lock(IpcMutex);
          C10_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
          ipcMemHandle_to_devptr.erase(handle);
        });
    std::weak_ptr<void> wp = sp;
    // To eliminate an additional search, we can use insert().
    // It doesn't overwrite when key already exists(ptr expired).
    // But in the deleter for sp we erased the entry,
    // this should be safe to do now.
    ipcMemHandle_to_devptr.insert(iter, {handle, wp});

    return sp;
  }
  std::string name() override {
    return "native";
  }
};

NativeCachingAllocator allocator;

void local_raw_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  allocator.free(ptr);
}

void setAllocatorSettings(const std::string& env) {
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace Native

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

// Size pretty-printer
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

namespace CudaMallocAsync {
// If this is put in its own header file, it gets incorrectly renamed in HIPify.
CUDAAllocator* allocator();

} // namespace CudaMallocAsync

struct BackendStaticInitializer {
  // Parses env for backend at load time, duplicating some logic from
  // CachingAllocatorConfig. CachingAllocatorConfig double-checks it later (at
  // runtime). Defers verbose exceptions and error checks, including Cuda
  // version checks, to CachingAllocatorConfig's runtime doublecheck. If this
  // works, maybe we should move all of CachingAllocatorConfig here?
  CUDAAllocator* parseEnvForBackend() {
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
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

std::atomic<CUDAAllocator*> allocator;
BackendStaticInitializer backend_static_initializer;

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
