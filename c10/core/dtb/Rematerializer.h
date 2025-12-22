#pragma once

#include <c10/core/dtb/comm_heads.h>

namespace c10{
namespace dtb{

// The rematerializer could be called to reinvoke an operator.
// Tensor point to remat which point to Tensor.
// To build the cycle remat support a default constructor,
// And allow you to fill in the member later.
struct Rematerializer : intrusive_ptr_target {
  rematerialize_function_t func;
  strongs inputs;
  weaks outputs;
  duration_t compute_cost;
  int64_t rid;   // remat func fingerprint
  // when some output in here get evicted, they should belong to this ecn.
  // a rematerializer have to track this,
  // because when multiple output of a rematerializer get evicted,
  // we only want to count the compute cost once.
  ecn_ptr ecn;
  
  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
                 duration_t compute_cost);

  Rematerializer(const Unsafe&,
                 const rematerialize_function_t& func,
                 const strongs& inputs,
                 int64_t rid,
                 duration_t compute_cost);

  void release_resources() final;
  void remat();
  void remat(int&);
  ecn_ptr get_ecn();
  CheckpointInfo get_cpi();
};

}
}