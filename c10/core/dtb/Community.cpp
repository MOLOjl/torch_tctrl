#include "Community.h"

namespace c10 {
namespace dtb {


using namespace std;

void SingletonCommunity::insert_node(const StrongDGNode& new_node, bool is_border) {
  auto it = border_marker.find(new_node);
  if(it!=border_marker.end()) {
    bool past_is_border = it->second;
    if(past_is_border==is_border) return;
    else erase_node(new_node);
  }
  if(is_border) {
    border_nodes.insert(new_node);
    border_marker[new_node] = true;
  } else {
    inner_nodes.insert(new_node);
    border_marker[new_node] = false;
  }
}

void SingletonCommunity::erase_node(const StrongDGNode& node) {
  auto it = border_marker.find(node);
  if(it!=border_marker.end()) {
    if(it->second) border_nodes.erase(node);
    else inner_nodes.erase(node);
  }
}

size_t SingletonCommunity::lock_borders() {
  // TODO: 增加剪枝条件
  size_t res = 0;
  if(!is_lock){
    int count = 0;
    for(auto& dgnode: border_nodes) {
      count++;
      if(count>DCR_LOCK_TOPS) break;
      dgnode->lock_value();
      res += dgnode->value.lock()->pool->memory;
    }
    is_lock = true;
  }
  return res;
}

void SingletonCommunity::unlock_borders() {
  if(is_lock) {
    for(auto& dgnode: border_nodes) {
      dgnode->unlock_value();
    }
    is_lock = false;
  }
}

void SingletonCommunity::clear_outers(const StrongDG& og, int cid) {
  for(auto it=inner_nodes.begin(); it!=inner_nodes.end(); ) {
    auto& nid = (*it)->nid;
    if(og->n2c[nid] != cid) {
      it = inner_nodes.erase(it);
      border_marker.erase((*it));
    }else{
      it++;
    }
  }
  for(auto it=border_nodes.begin(); it!=border_nodes.end(); ) {
    auto& nid = (*it)->nid;
    if(og->n2c[nid] != cid) {
      it = border_nodes.erase(it);
      border_marker.erase((*it));
    }else{
      it++;
    }
  }
}

Community::Community(DynamicGraph gc, int nbp, double minm) : nb_pass(nbp), min_modularity(minm){
  g = gc;
  size = g.nb_nodes;

  neigh_weight.resize(size,-1);
  neigh_pos.resize(size);
  neigh_last=0;

  n2c.resize(size);
  in.resize(size);
  tot.resize(size);

  for (int i=0 ; i<size ; i++) {
    n2c[i] = i;  // 初始化每个结点为一个社区
    in[i]  = g.nb_selfloops(i);
    tot[i] = g.weighted_degree(i);
  }

  g.n2c = n2c;

}


void Community::insert_edge(nid_t start, nid_t end, const weak& s_cell, const weak& e_cell, const StrongDG& og, float w) {
  // 执行前必须首先对og进行边的插入
  TORCH_INTERNAL_ASSERT(start < og->n2c.size())
  TORCH_INTERNAL_ASSERT(end   < og->n2c.size())

  auto s_c = og->n2c[start];
  auto e_c = og->n2c[end];

  auto graph_add = [&](const int& snode, const int& enode){
    // 限制新增节点的编号为新增单例社区，并插入到comm graph中， nb_nodes实际上比Node数目多1个
    auto renum_snode = snode < g.nb_nodes ? snode : g.nb_nodes;
    auto renum_enode = enode < g.nb_nodes ? enode : g.nb_nodes;
    renum_enode = (renum_enode==renum_snode)&&(snode!=enode) ? renum_snode+1 : renum_enode;
    // 更新原图中的社区归属
    og->n2c[start] = renum_snode;
    og->n2c[end] = renum_enode;
    // 插入社区图节点
    // cerr << "before insert" << (renum_snode) << "-" << renum_enode << " total_weight:" << g.total_weight << endl;
    g.insert_edge(renum_snode, renum_enode, s_cell, e_cell, w);
    // cerr << "after insert" << (renum_snode) << "-" << renum_enode << " total_weight:" << g.total_weight << endl;
    return std::pair<int, int>{renum_snode, renum_enode};
  };

  auto c_se_pair = graph_add(s_c, e_c);

  /**
   * 这里接收的node是一个社区图中的node，原图的node id不能直接作为Community中的node使用
   * @param neighbors: node's neighbors in comm graph
   * @param node: node id in comm graph(comm id)
  */
  auto community_add = [&](const int& node, const int comm){
    
    if(node >= size){
      /// 新增为单例社区
      insert_new(node, comm, g.nb_neighbors(node)); // 无向图这一项即为邻居的个数
    }else{
      insert(node, comm, g.nb_neighbors(node));
    }

    /// TODO: 下面这一部分是有待检验的，目前仅将新节点作为单例社区
    /// 新增的节点如果没有关联边，则作为单例社区
    // if(neigbors.empty()){
    //   insert_new(node, node, 0.);     // 单例社区内部权重为0
    //   return;
    // }
    // /// 新增的节点如果所有的边都关联到同一社区，则设置其归属到对应社区
    // bool all_same_comm = true;
    // nid_t last_comm = neigbors[0];
    // for(const auto& neigh: neigbors){
    //   if(og->n2c[neigh]!=last_comm){
    //     all_same_comm = false;
    //     break;
    //   }
    // }
    // if(all_same_comm){
    //   og->n2c[start] = last_comm;
    //   insert(node, last_comm, g.weighted_degree(node));
    // }
    /// 如果关联到了不同社区，设置其归属到边权重连接最大的社区（隐患：对于无权重图，和哪个社区连接多就取哪个，是否会导致某一社区不断扩张，可能效果不如直接连接为单例社区？
    ////// 1: 直接连接到边权重最大的社区
    ////// 2: 关联到最大社区
  };
  community_add(c_se_pair.first, g.n2c[c_se_pair.first]);
  community_add(c_se_pair.second, g.n2c[c_se_pair.second]);
}

/**
 * @param og: og maintain the orginal nodes and their community id
 */
void Community::add_from_edges(char *filename, int start_batch, int end_batch, int type, DynamicGraph* og){

  // 完成原始图的添加，单例社区的初始化
  // og->add_from_edges(filename, start_batch, end_batch, type);

  // // 打开文件
  // std::ifstream file(filename);
  
  // // 检查文件是否成功打开
  // if (!file) {
  //     std::cerr << "无法打开文件" << std::endl;
  //     return;
  // }
  
  // std::string line;
  // int cur_line_num = 0;
  // // 逐行读取
  // while (getline(file, line)) {
  //   cur_line_num++;
  //   if(cur_line_num < start_batch) continue;
  //   if(end_batch != -1 && cur_line_num >= end_batch) break;
  //   std::istringstream iss(line);
  //   int start, end, w;
    
  //   if(type==UNWEIGHTED){
  //     if (iss >> start >> end) {
  //       auto s_c = og->n2c[start];
  //       auto e_c = og->n2c[end];
  //       // auto neigh_start = og->neighbors(start).first;
  //       // auto neigh_end = og->neighbors(end).first;
  //       // cerr << start << "(" << s_c << ")-" << end << "(" << e_c << ") " 
  //       //      << n2c.size() << " " << g.n2c.size() << endl;
  //           // << " cs:" <<  c_se_pair.first <<" ce:" << c_se_pair.second << " trigger\n";

  //       auto graph_add = [&](const int& snode, const int& enode){
  //         // 限制新增节点的编号为新增单例社区，并插入到comm graph中， nb_nodes实际上比Node数目多1个
  //         auto renum_snode = snode < g.nb_nodes ? snode : g.nb_nodes;
  //         auto renum_enode = enode < g.nb_nodes ? enode : g.nb_nodes;
  //         renum_enode = (renum_enode==renum_snode)&&(snode!=enode) ? renum_snode+1 : renum_enode;
  //         // 更新原图中的社区归属
  //         og->n2c[start] = renum_snode;
  //         og->n2c[end] = renum_enode;
  //         // 插入社区图节点
  //         // cerr << "before insert" << (renum_snode) << "-" << renum_enode << " total_weight:" << g.total_weight << endl;
  //         g.insert_edge(renum_snode, renum_enode);
  //         // cerr << "after insert" << (renum_snode) << "-" << renum_enode << " total_weight:" << g.total_weight << endl;
  //         return std::pair<int, int>{renum_snode, renum_enode};
  //       };

  //       auto c_se_pair = graph_add(s_c, e_c);

  //       /**
  //        * 这里接收的node是一个社区图中的node，原图的node id不能直接作为Community中的node使用
  //        * @param neighbors: node's neighbors in comm graph
  //        * @param node: node id in comm graph(comm id)
  //       */
  //       auto community_add = [&](const int& node, const int comm){
          
  //         if(node >= size){
  //           /// 新增为单例社区
  //           insert_new(node, comm, g.nb_neighbors(node)); // 无向图这一项即为邻居的个数
  //         }else{
  //           insert(node, comm, g.nb_neighbors(node));
  //         }

  //         /// 新增的节点如果没有关联边，则作为单例社区
  //         // if(neigbors.empty()){
  //         //   insert_new(node, node, 0.);     // 单例社区内部权重为0
  //         //   return;
  //         // }
  //         // /// 新增的节点如果所有的边都关联到同一社区，则设置其归属到对应社区
  //         // bool all_same_comm = true;
  //         // nid_t last_comm = neigbors[0];
  //         // for(const auto& neigh: neigbors){
  //         //   if(og->n2c[neigh]!=last_comm){
  //         //     all_same_comm = false;
  //         //     break;
  //         //   }
  //         // }
  //         // if(all_same_comm){
  //         //   og->n2c[start] = last_comm;
  //         //   insert(node, last_comm, g.weighted_degree(node));
  //         // }
  //         /// 如果关联到了不同社区，设置其归属到边权重连接最大的社区（隐患：对于无权重图，和哪个社区连接多就取哪个，是否会导致某一社区不断扩张，可能效果不如直接连接为单例社区？
  //         ////// 1: 直接连接到边权重最大的社区
  //         ////// 2: 关联到最大社区
  //       };
  //       community_add(c_se_pair.first, g.n2c[c_se_pair.first]);
  //       community_add(c_se_pair.second, g.n2c[c_se_pair.second]);
  //     }
  //   }else{
  //     if (iss >> start >> end >> w) {
  //       auto s_c = og->n2c[start];
  //       auto e_c = og->n2c[end];
  //       auto neigh_start = og->neighbors(start).first;
  //       auto neigh_end = og->neighbors(end).first;

  //       auto graph_add = [&](const int& snode, const int& enode){
  //         // 限制新增节点的编号为新增单例社区，并插入到comm graph中
  //         auto renum_snode = snode < n2c.back() ? snode : n2c.back()+1;
  //         auto renum_enode = enode < n2c.back() ? enode : n2c.back()+1;
  //         renum_enode = renum_enode==renum_snode ? renum_enode+1 : renum_enode;
  //         // 更新原图中的社区归属
  //         og->n2c[start] = renum_snode;
  //         og->n2c[end] = renum_enode;
  //         // 插入社区图节点
  //         g.insert_edge(renum_snode, renum_enode, w);
  //         g.insert_edge(renum_enode, renum_snode, w);
  //       };

  //       graph_add(s_c, e_c);
  //     }
  //   }
  // }
  
  // // 关闭文件
  // file.close();
}

void
Community::display() {
  for (int i=0 ; i<size ; i++)
    cerr << " " << i << "/" << n2c[i] << "/" << in[i] << "/" << tot[i] ;
  cerr << endl;
}


double Community::modularity() {
  double q  = 0.;
  double m2 = (double)g.total_weight;
  // printf("modularity calculate:%lf, \n", m2);

  for (int i=0 ; i<size ; i++) {
    if (tot[i]>0)
      q += (double)in[i]/m2 - ((double)tot[i]/m2)*((double)tot[i]/m2);
  }

  return q;
}

void Community::neigh_comm(unsigned int node) {
  for (unsigned int i=0 ; i<neigh_last ; i++)   // 清空记录值
    neigh_weight[neigh_pos[i]]=-1;
  neigh_last=0;

  // 获取node的邻居
  auto p = g.neighbors(node);

  unsigned int deg = g.nb_neighbors(node);

  neigh_pos[0]=n2c[node];
  neigh_weight[neigh_pos[0]]=0;
  neigh_last=1;

  // 标记邻居neigh所属社区及对应link的权重
  for (unsigned int i=0 ; i<deg ; i++) {
    unsigned int neigh        = p.first[i];
    unsigned int neigh_comm   = n2c[neigh];
    double neigh_w = p.second[i];
    
    if (neigh!=node) {
      if (neigh_weight[neigh_comm]==-1) {
        neigh_weight[neigh_comm]=0.;
        neigh_pos[neigh_last++]=neigh_comm;   // 标记邻居所属社区id
      }
      neigh_weight[neigh_comm]+=neigh_w;      // 累积到该社区的权重
    }
  }
}

void
Community::partition2graph() {
  vector<int> renumber(size, -1);
  for (int node=0 ; node<size ; node++) {
    renumber[n2c[node]]++;
  }

  int final=0;
  for (int i=0 ; i<size ; i++)
    if (renumber[i]!=-1)
      renumber[i]=final++;


  for (int i=0 ; i<size ; i++) {
    auto p = g.neighbors(i);

    int deg = g.nb_neighbors(i);
    for (int j=0 ; j<deg ; j++) {
      int neigh = p.first[j];
      cout << renumber[n2c[i]] << " " << renumber[n2c[neigh]] << endl;
    }
  }
}

void
Community::display_partition() {
  vector<int> renumber(size, -1);
  for (int node=0 ; node<size ; node++) {
    renumber[n2c[node]]++;
  }

  int final=0;
  for (int i=0 ; i<size ; i++)
    if (renumber[i]!=-1)
      renumber[i]=final++;

  for (int i=0 ; i<size ; i++)
    cout << i << " " << renumber[n2c[i]] << endl;
}

nid_t max_c(vector<nid_t> n2c){
  nid_t res = 0;
  for(auto t: n2c){
    res = max(res, t);
  }
  return res;
}

/**
 * @param og: og maintain the orginal nodes and their community id
 * 将新的社区关系转换成一个新的DynamicGraph，并维护og的社区关系变更
 */
DynamicGraph Community::partition2graph_binary(const StrongDG& og) {
  // Renumber communities
  vector<int> renumber(size, -1);   // 同一社区id归约
  for (int node=0 ; node<size ; node++) {
    renumber[n2c[node]]++;
  }

  int idx=0;    // 对社区id进行重编号
  for (int i=0 ; i<size ; i++)
    if (renumber[i]!=-1)
      renumber[i]=idx++;

  // Compute communities
  vector<vector<int> > comm_nodes(idx);   // 共有idx个社区|新结点
  for (int node=0 ; node<size ; node++) { // 每个社区用一个vector存储其成员节点, renumber[n2c[node]]为重编号后的社区id
    comm_nodes[renumber[n2c[node]]].push_back(node);
  }
  // flush original community info
  for(nid_t i=0; i<og->nb_nodes; i++){
    if(og->n2c[i] >= n2c.size()){       // 空结点并不会被归约到，直接将其指定到前一个社区
      og->n2c[i] = renumber[n2c[og->n2c[i-1]]];
      // printf("og %ld-th: %d-%d, out of sizes(%d), determine to comm %d.\n", i, og->n2c[i], n2c.size(), size, og->n2c[i]);
    }else{
      og->n2c[i] = renumber[n2c[og->n2c[i]]];
    }
  }

  // Compute weighted graph
  DynamicGraph g2;
  g2.nb_nodes = comm_nodes.size();
  g2.degrees.resize(comm_nodes.size());
  g2.edges.resize(comm_nodes.size());
  // if(og->is_weighted_graph())
  g2.weights.resize(comm_nodes.size());
  g2.total_weight = 0;
  // g2.n2c = renumber;

  int comm_deg = comm_nodes.size();
  for (int comm=0 ; comm<comm_deg ; comm++) {
    map<int,float> m;               // 记录到邻居社区的权重
    map<int,float>::iterator it;

    int comm_size = comm_nodes[comm].size();
    for (int node=0 ; node<comm_size ; node++) {            // 社区间权重和并
      auto p = g.neighbors(comm_nodes[comm][node]);
      int deg = g.nb_neighbors(comm_nodes[comm][node]);
      for (int i=0 ; i<deg ; i++) {
        int neigh        = p.first[i];
        int neigh_comm   = renumber[n2c[neigh]];
        double neigh_weight = p.second[i];

        it = m.find(neigh_comm);
        if (it==m.end())
          m.insert(make_pair(neigh_comm, neigh_weight));
        else
          it->second += neigh_weight;
      }
    }
    //    deg(0)=degrees[0]
    //    deg(k)=degrees[k]-degrees[k-1]
    g2.degrees[comm]= (comm==0) ? m.size() : g2.degrees[comm-1]+m.size();
    g2.nb_edges += m.size();

        
    for (it = m.begin() ; it!=m.end() ; it++) {
      g2.total_weight += it->second;
      g2.links.push_back(it->first);
      g2.edges[comm].push_back(it->first);
      g2.weights[comm].push_back(it->second);
    }
  }

  return g2;
}


bool
Community::one_level() {
  bool improvement=false ;
  int nb_moves;
  int nb_pass_done = 0;
  double new_mod   = modularity();
  double cur_mod   = new_mod;

  // 原地打乱
  vector<int> random_order(size);
  for (int i=0 ; i<size ; i++)
    random_order[i]=i;
  for (int i=0 ; i<size-1 ; i++) {
    int rand_pos = rand()%(size-i)+i;
    int tmp      = random_order[i];
    random_order[i] = random_order[rand_pos];
    random_order[rand_pos] = tmp;
  }

  // repeat while 
  //   there is an improvement of modularity
  //   or there is an improvement of modularity greater than a given epsilon 
  //   or a predefined number of pass have been done
  do {
    cur_mod = new_mod;
    nb_moves = 0;
    nb_pass_done++;

    // for each node: remove the node from its community and insert it in the best community
    for (int node_tmp=0 ; node_tmp<size ; node_tmp++) {
//      int node = node_tmp;
      int node = random_order[node_tmp];
      int node_comm     = n2c[node];
      double w_degree = g.weighted_degree(node);

      // computation of all neighboring communities of current node
      neigh_comm(node);
      // remove node from its current community
      remove(node, node_comm, neigh_weight[node_comm]);

      // compute the nearest community for node
      // default choice for future insertion is the former community
      int best_comm        = node_comm;
      double best_nblinks  = 0.;  // 默认权重都为1时，这个值就是和该社区的link数目，有权重就是加权link
      double best_increase = 0.;
      for (unsigned int i=0 ; i<neigh_last ; i++) {
        double increase = modularity_gain(node, neigh_pos[i], neigh_weight[neigh_pos[i]], w_degree);
        if (increase > best_increase) {
          best_comm     = neigh_pos[i];
          best_nblinks  = neigh_weight[neigh_pos[i]];
          best_increase = increase;
        }
      }

      // insert node in the nearest community
      insert(node, best_comm, best_nblinks);
     
      if (best_comm!=node_comm)
        nb_moves++;
    }

    double total_tot=0;
    double total_in=0;
    for (unsigned int i=0 ; i<tot.size() ;i++) {
      total_tot+=tot[i];
      total_in+=in[i];
    }

    new_mod = modularity();
    if (nb_moves>0)
      improvement=true;
    
  } while (nb_moves>0 && new_mod-cur_mod>min_modularity);

  return improvement;
}

}
}

