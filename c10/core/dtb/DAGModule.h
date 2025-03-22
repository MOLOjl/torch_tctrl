#pragma once

#include <vector>
#include <string>
#include <limits>
#include <queue>
#include <unordered_map>
#include <chrono>
#include <future>
#include <c10/core/dtb/comm_heads.h>
#include <c10/core/dtb/CheckpointTensorCell.h>

namespace c10 {
namespace dtb {

struct DAGNode;
using SDAGNode = intrusive_ptr<DAGNode>;
// using WeakDAGNode = weak_intrusive_ptr<DAGNode>;
using dag_nid_t = long;
using timestamp_t = std::chrono::time_point<std::chrono::steady_clock>;

inline timestamp_t get_current_time() { return std::chrono::steady_clock::now(); }

struct DAGNode : intrusive_ptr_target {
private:
    bool is_lock = false;
    
public:
    dag_nid_t nid;
    int distance;
    std::vector<std::pair<SDAGNode, int>> out_nodes;
    std::vector<SDAGNode> in_nodes;
    // 这几个部分与weak所记录的可能有些重叠，为了集中处理逻辑，在这里再记录一次
    int in_degree;
    int out_degree;
    int level;
    weak cptc;

    DAGNode(dag_nid_t nid, const weak& cptc);
    std::string to_string() const;
    void lock_node();
    void unlock_node();
    void release_resources() override;
};


struct DynamicDAGShortestPath : intrusive_ptr_target {
private:
    SDAGNode start_node;
    dag_nid_t start_nid;
    std::queue<SDAGNode> queue;
    std::unordered_map<int, SDAGNode> distance_to_max_level_node;

    int last_same_idx = 0;
    int operation_counter = 0;
    // distance的最近访问时间戳
    std::unordered_map<int, timestamp_t> distance_to_last_change_time;
    std::vector<SDAGNode> last_timer_order_nodes;

    std::future<void> stable_window_future;
    std::mutex distance_mutex; 

    void _insert_sorted(const SDAGNode& node);
    void _update_sorted_nodes(const SDAGNode& node);

    inline void _update_distance_timestamp(int distance, const timestamp_t& time) {
        std::lock_guard<std::mutex> lock(distance_mutex);
        distance_to_last_change_time[distance] = time;
    }

public:
    void _update_stable_window(bool final=false);
    int total_lock_counts = 0;
    int total_unlock_counts = 0;
    std::unordered_map<dag_nid_t, SDAGNode> nodes;
    std::vector<SDAGNode> sorted_nodes;

    DynamicDAGShortestPath(dag_nid_t nid, const weak& cptc);
    void add_node(dag_nid_t nid, const weak& cptc);
    void add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight);
    int get_shortest_distance(dag_nid_t nid);
    void relax(const SDAGNode& u, const SDAGNode& v, int weight);
    void process_queue();
    std::vector<SDAGNode> get_sorted_nodes();
    void wait_async_task();
    void clear_all_nodes();
    void release_resources() override;
};

using SDAG = intrusive_ptr<DynamicDAGShortestPath>;

struct MultiDAGShortestPaths : intrusive_ptr_target {
private:
    int bid;

public:
    std::vector<SDAG> subgraphs;
    std::unordered_map<dag_nid_t, SDAG> node_to_subgraph;

    explicit MultiDAGShortestPaths(int batch_id);
    void add_node(dag_nid_t nid, const weak& cptc);
    void add_edge(dag_nid_t s_id, dag_nid_t t_id, const weak& s, const weak& t, int weight);
    int get_shortest_distance(dag_nid_t nid);
    std::vector<SDAGNode> get_sorted_nodes_for_subgraph_containing(dag_nid_t nid);
    void clear_all_graphs();
    void release_resources() override;
};

using SMultiDAG = intrusive_ptr<MultiDAGShortestPaths>;


}
}
