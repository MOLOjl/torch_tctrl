#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>

namespace at { namespace native {

inline CheckpointTensorImpl* get_sparse_impl(const Tensor& self) {
  TORCH_INTERNAL_ASSERT(
      self.is_checkpoint(), "_internal_get_CheckpointTensorImpl: not a checkpoint tensor");
  return static_cast<CheckpointTensorImpl*>(self.unsafeGetTensorImpl());
}

// Tensor checkpoint_add(const Tensor& a, const Tensor& b, const c10::Scalar& c) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::add(vec.at(0), vec.at(1), c)};
//     };
//   return CheckpointTensorImpl::make("add", rt, {a, b})[0];
// }

// Tensor checkpoint_t(at::Tensor const& a) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::t(vec.at(0))};
//     };
//   return CheckpointTensorImpl::make("t", rt, {a})[0];
// }

// Tensor checkpoint_add(at::Tensor const& a, const Scalar& b, const Scalar& c) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::add(vec.at(0), b, c)};
//     };
//   return CheckpointTensorImpl::make("add", rt, {a})[0];
// }

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, const Scalar& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).add_(vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("add_", mt, {a, b}, {0});
  return a;
}

// Tensor checkpoint_mul(at::Tensor const& a, at::Tensor const& b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::mul(vec.at(0), vec.at(1))};
//     };
//   return CheckpointTensorImpl::make("mul", rt, {a, b})[0];
// }

Tensor& checkpoint_mul_(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("mul_", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_mul_(at::Tensor& a, const Scalar& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).mul_(b);
    };
  CheckpointTensorImpl::mutate("mul_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_zeros_like(at::Tensor const& a, c10::TensorOptions const& b, c10::optional<c10::MemoryFormat> c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::zeros_like(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("zeros_like", rt, {a})[0];
}

Tensor checkpoint_ones_like(at::Tensor const& a, c10::TensorOptions const& b, c10::optional<c10::MemoryFormat> c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones_like(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("ones_like", rt, {a})[0];
}

Tensor checkpoint_addcmul(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addcmul(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("addcmul", rt, {a, b, c})[0];
}

Tensor& checkpoint_addcmul_(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).addcmul_(vec.at(1), vec.at(2), d);
    };
  CheckpointTensorImpl::mutate("addcmul_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_abs(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::abs(vec.at(0))};
    };
  return CheckpointTensorImpl::make("abs", rt, {a})[0];
}

Tensor checkpoint_sqrt(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sqrt(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sqrt", rt, {a})[0];
}

Tensor& checkpoint_addcdiv_(at::Tensor& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).addcdiv_(vec.at(1), vec.at(2), d);
    };
  CheckpointTensorImpl::mutate("addcdiv_", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_addcdiv(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, c10::Scalar d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addcdiv(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  return CheckpointTensorImpl::make("addcdiv", rt, {a, b, c})[0];
}

Tensor checkpoint_to(at::Tensor const& a, c10::TensorOptions const& b, bool c, bool d, c10::optional<c10::MemoryFormat> e) {
  c10::TensorOptions b_ = b;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).to(b_, c, d, e)};
    };
  return CheckpointTensorImpl::make("to", rt, {a})[0];
}

// Tensor checkpoint_div(const Tensor& a, const Tensor& b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::div(vec.at(0), vec.at(1))};
//     };
//   return CheckpointTensorImpl::make("div", rt, {a, b})[0];
// }

Tensor& checkpoint_div_(Tensor& a, const Tensor& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).div_(vec.at(1));
    };
  CheckpointTensorImpl::mutate("div_", mt, {a, b}, {0});
  return a;
}

// Tensor checkpoint_clone(at::Tensor const& a, c10::optional<c10::MemoryFormat> b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::clone(vec.at(0), b)};
//   };
//   return CheckpointTensorImpl::make("clone", rt, {a})[0];
// }

// Tensor checkpoint_where(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::where(vec.at(0), vec.at(1), vec.at(2))};
//     };
//   return CheckpointTensorImpl::make("where", rt, {a, b, c})[0];
// }

Tensor checkpoint_constant_pad_nd(Tensor const& a, c10::ArrayRef<long> b, c10::Scalar const& c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::constant_pad_nd(vec.at(0), b_, c)};
    };
  return CheckpointTensorImpl::make("constant_pad_nd", rt, {a})[0];
}

Tensor checkpoint_binary_cross_entropy(const Tensor& a, const Tensor& b, const c10::optional<Tensor>& c, long d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy(vec.at(0), vec.at(1), vec.at(2), d)};
    };
  c10::MaybeOwned<Tensor> c_maybe_owned = at::borrow_from_optional_tensor(c);
  const Tensor& c_ = *c_maybe_owned;
  return CheckpointTensorImpl::make("binary_cross_entropy", rt, {a, b, c_})[0];
}

Tensor& checkpoint_binary_cross_entropy_out(const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor input = vec.at(0), loss = vec.at(3);
      at::binary_cross_entropy_outf(input, vec.at(1), vec.at(2), reduction, loss);
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& w_ = *weight_maybe_owned;
  CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {input, target, w_, loss}, {0});
  return loss;
}

Tensor checkpoint_binary_cross_entropy_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e) { 
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::binary_cross_entropy_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e)};
    };
  return CheckpointTensorImpl::make("binary_cross_entropy_backward", rt, {a, b, c, d})[0];
}

// inline at::Tensor & binary_cross_entropy_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input)
Tensor& checkpoint_binary_cross_entropy_backward_out(const Tensor& grad, const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor& grad_input) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor grad = vec.at(0), grad_input = vec.at(4);
      at::binary_cross_entropy_backward_outf(grad, vec.at(1), vec.at(2), vec.at(3), reduction, grad_input);
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& w_ = *weight_maybe_owned;
  CheckpointTensorImpl::mutate("binary_cross_entropy_backward_out", mt, {grad, input, target, w_, grad_input}, {0});
  return grad_input;
}

// Tensor checkpoint_embedding(const Tensor& a, const Tensor& b, long c, bool d, bool e) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::embedding(vec.at(0), vec.at(1), c, d, e)};
//     };
//   return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
// }

Tensor checkpoint_embedding_backward(const Tensor& a, const Tensor& b, long c, long d, bool e, bool f) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_backward(vec.at(0), vec.at(1), c, d, e, f)};
    };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
checkpoint_cudnn_batch_norm(const Tensor& a, const Tensor& b, const c10::optional<Tensor>& c, const c10::optional<Tensor>& d, const c10::optional<Tensor>& e, bool f, double g, double h) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, h);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
    };
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(c);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = c10::value_or_else(d, [] {return Tensor();});
  const Tensor& running_var_t = c10::value_or_else(e, [] {return Tensor();});
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm", rt, {a, b, bias_t, running_mean_t, running_var_t});
  return {ret[0], ret[1], ret[2], ret[3]};
}

/**
  const Tensor& input_t,
  const Tensor& grad_output_t,
  const Tensor& weight_t,
  // Unused: but we require them to be passed so that double backwards
  // has access
  const c10::optional<Tensor>& running_mean_opt,
  const c10::optional<Tensor>& running_var_opt,
  const c10::optional<Tensor>& save_mean_t_opt,
  const c10::optional<Tensor>& save_var_t_opt,
  double epsilon,
  const Tensor& reserveSpace
*/
std::tuple<Tensor, Tensor, Tensor> checkpoint_cudnn_batch_norm_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, 
  const c10::optional<Tensor>& d, const c10::optional<Tensor>& e, const c10::optional<Tensor>& f, const c10::optional<Tensor>& g, double h, at::Tensor const& i) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::cudnn_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), h, vec.at(7));
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  const Tensor& d_ = c10::value_or_else(d, [] {return Tensor();});
  const Tensor& e_ = c10::value_or_else(e, [] {return Tensor();});
  const Tensor& f_ = c10::value_or_else(f, [] {return Tensor();});
  const Tensor& g_ = c10::value_or_else(g, [] {return Tensor();});
  auto ret = CheckpointTensorImpl::make("cudnn_batch_norm_backward", rt, {a, b, c, d_, e_, f_, g_, i});
  return {ret[0], ret[1], ret[2]};
}

// Tensor checkpoint_as_strided(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::optional<long> d) {
//   std::vector<long> b_ = b.vec(), c_ = c.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::as_strided(vec.at(0), b_, c_, d)};
//     };
//   return CheckpointTensorImpl::make("as_strided", rt, {a})[0];
// }

Tensor checkpoint__masked_scale(const Tensor& a, const Tensor& b, double c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_masked_scale(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("_masked_scale", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h, bool i) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution(vec.at(0), vec.at(1), c_, d_, e_, f, g, h, i)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution", rt, {a, b})[0];
}
// const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32
Tensor checkpoint_cudnn_convolution_transpose(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, bool j) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cudnn_convolution_transpose(vec.at(0), vec.at(1), c_, d_, e_, f_, g, h, i, j)};
    };
  return CheckpointTensorImpl::make("cudnn_convolution_transpose", rt, {a, b})[0];
}

// std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, bool j, std::array<bool, 2ul> k) {
//   std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       auto ret = at::cudnn_convolution_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g, h, i, j, k);
//       return {std::get<0>(ret), std::get<1>(ret)};
//     };
//   auto ret = CheckpointTensorImpl::make("cudnn_convolution_backward", rt, {a, b, c});
//   return {ret[0], ret[1]};
// }

// std::tuple<Tensor, Tensor> checkpoint_cudnn_convolution_transpose_backward(const Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, long h, bool i, bool j, bool k, std::array<bool, 2ul> l) {
//   std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       auto ret = at::cudnn_convolution_transpose_backward(vec.at(0), vec.at(1), vec.at(2), d_, e_, f_, g_, h, i, j, k, l);
//       return {std::get<0>(ret), std::get<1>(ret)};
//     };
//   auto ret = CheckpointTensorImpl::make("cudnn_convolution_transpose_backward", rt, {a, b, c});
//   return {ret[0], ret[1]};
// }

// Tensor checkpoint_cudnn_convolution_backward_input(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, bool j) {
//   std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::cudnn_convolution_backward_input(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i, j)};
//     };
//   return CheckpointTensorImpl::make("cudnn_convolution_backward_input", rt, {b, c})[0];
// }

// Tensor checkpoint_cudnn_convolution_transpose_backward_input(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, long f, bool g, bool h, bool i) {
//   std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::cudnn_convolution_transpose_backward_input(vec.at(0), vec.at(1), c_, d_, e_, f, g, h, i)};
//     };
//   return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_input", rt, {a, b})[0];
// }

// Tensor checkpoint_cudnn_convolution_backward_weight(IntArrayRef a, const Tensor & b, const Tensor & c, IntArrayRef d, IntArrayRef e, IntArrayRef f, int64_t g, bool h, bool i, bool j) {
//   std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::cudnn_convolution_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i, j)};
//     };
//   return CheckpointTensorImpl::make("cudnn_convolution_backward_weight", rt, {b, c})[0];
// }

// Tensor checkpoint_cudnn_convolution_transpose_backward_weight(c10::ArrayRef<long> a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, long g, bool h, bool i, bool j) {
//   std::vector<long> a_ = a.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::cudnn_convolution_transpose_backward_weight(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i, j)};
//     };
//   return CheckpointTensorImpl::make("cudnn_convolution_transpose_backward_weight", rt, {b, c})[0];
// }

Tensor checkpoint_relu(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::relu(vec.at(0))};
    };
  auto res = CheckpointTensorImpl::make("relu", rt, {a})[0];
  return res;
}

Tensor& checkpoint_relu_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).relu_();
    };
  CheckpointTensorImpl::mutate("relu_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_log(at::Tensor const& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::log(vec.at(0))};
    };
  return CheckpointTensorImpl::make("log", rt, {a})[0];
}

Tensor& checkpoint_log_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::log_out(a_, vec.at(1));
    };
  CheckpointTensorImpl::mutate("log_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_rsub(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("rsub", rt, {a, b})[0];
}

Tensor checkpoint_rsub(at::Tensor const& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("rsub", rt, {a})[0];
}

// Tensor checkpoint_mul(at::Tensor const& a, c10::Scalar b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::mul(vec.at(0), b)};
//     };
//   return CheckpointTensorImpl::make("mul", rt, {a})[0];
// }

// std::tuple<Tensor&, Tensor&> checkpoint_max_pool2d_with_indices_out(Tensor& a, Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h) {
//   std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0), b_ = vec.at(1);
//       at::max_pool2d_with_indices_out(a_, b_, vec.at(2), d_, e_, f_, g_, h);
//     };
//   CheckpointTensorImpl::mutate("max_pool2d_with_indices_out", mt, {a, b, c}, {0, 1});
//   return {a, b};
// }

Tensor checkpoint_avg_pool2d(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, bool e, bool f, c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d(vec.at(0), b_, c_, d_, e, f, g)};
    };
  return CheckpointTensorImpl::make("avg_pool2d", rt, {a})[0];
}

Tensor checkpoint_avg_pool2d_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::avg_pool2d_backward(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
    };
  return CheckpointTensorImpl::make("avg_pool2d_backward", rt, {a, b})[0];
}

// Tensor& checkpoint_avg_pool2d_out(Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f, bool g, c10::optional<long> h) {
//   std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0);
//       at::avg_pool2d_out(a_, vec.at(1), c_, d_, e_, f, g, h);
//     };
//   CheckpointTensorImpl::mutate("avg_pool2d_out", mt, {a, b}, {0});
//   return a;
// }

Tensor& checkpoint_avg_pool2d_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, bool h, c10::optional<long> i) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::avg_pool2d_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, i);
    };
  CheckpointTensorImpl::mutate("avg_pool2d_backward_grad_input", mt, {a, b, c}, {0});
  return a;
}

// std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(const Tensor& a, c10::ArrayRef<long> b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, bool f) {
//   std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       auto ret = at::max_pool2d_with_indices(vec.at(0), b_, c_, d_, e_, f);
//       return {std::get<0>(ret), std::get<1>(ret)};
//     };
//   auto ret = CheckpointTensorImpl::make("max_pool2d_backward", rt, {a});
//   return {ret[0], ret[1]};
// }

// Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h, const Tensor& i) {
//   std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0);
//       at::max_pool2d_with_indices_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, vec.at(3));
//     };
//   CheckpointTensorImpl::mutate("max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
//   return a;
// }

// Tensor checkpoint_max_pool2d_with_indices_backward(const Tensor& a, const Tensor& b, c10::ArrayRef<long> c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, bool g, const Tensor& h) {
//   std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::max_pool2d_with_indices_backward(vec.at(0), vec.at(1), c_, d_, e_, f_, g, vec.at(2))};
//     };
//   return CheckpointTensorImpl::make("max_pool2d_with_indices_backward", rt, {a, b, h})[0];
// }

Tensor checkpoint_view(const Tensor& a, c10::ArrayRef<long> b) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).view(b_)};
    };
  return CheckpointTensorImpl::make("view", rt, {a})[0];
}

Tensor checkpoint_ne_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("ne_Scalar", rt, {a})[0];
}

Tensor& checkpoint_ne_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("ne_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_ne_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("ne_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_ne_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::ne_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("ne_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_eq_Scalar(const Tensor& a, c10::Scalar b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), b)};
    };
  return CheckpointTensorImpl::make("eq_Scalar", rt, {a})[0];
}

Tensor& checkpoint_eq_Scalar_out(Tensor& a, const Tensor& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), c);
    };
  CheckpointTensorImpl::mutate("eq_Scalar_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_eq_Tensor(const Tensor& a, const Tensor& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("eq_Tensor", rt, {a, b})[0];
}

Tensor& checkpoint_eq_Tensor_out(Tensor& a, const Tensor& b, const Tensor& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::eq_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("eq_Tensor_out", mt, {a, b, c}, {0});
  return a;
}

Tensor checkpoint_addmm(const Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addmm(vec.at(0), vec.at(1), vec.at(2), d, e)};
    };
  return CheckpointTensorImpl::make("addmm", rt, {a, b, c})[0];
}

Tensor& checkpoint_addmm_out(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, c10::Scalar e, c10::Scalar f) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::addmm_out(a_, vec.at(1), vec.at(2), d, e, f);
    };
  CheckpointTensorImpl::mutate("addmm_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_addmm_(Tensor& a, const Tensor& b, const Tensor& c, c10::Scalar d, c10::Scalar e) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.addmm_(vec.at(1), vec.at(2), d, e);
    };
  CheckpointTensorImpl::mutate("addmm_", mt, {a, b, c}, {0});
  return a;
}

// Tensor checkpoint_sigmoid(const Tensor& a) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::sigmoid(vec.at(0))};
//     };
//   return CheckpointTensorImpl::make("sigmoid", rt, {a})[0];
// }

Tensor& checkpoint_sigmoid_(Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      a.sigmoid_();
    };
  CheckpointTensorImpl::mutate("sigmoid_", mt, {a}, {0});
  return a;
}

Tensor checkpoint__log_softmax(const Tensor& a, long b, bool c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("_log_softmax", rt, {a})[0];
}

// const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, const at::Tensor & out
Tensor checkpoint__log_softmax_backward_data(const Tensor& a, const Tensor& b, long c, c10::ScalarType d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_log_softmax_backward_data(vec.at(0), vec.at(1), c, d)};
    };
  return CheckpointTensorImpl::make("_log_softmax_backward_data", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss_forward(const Tensor& a, const Tensor& b, const Tensor& c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::nll_loss_forward(vec.at(0), vec.at(1), vec.at(2), d, e);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("nll_loss_forward", rt, {a, b, c});
  return {ret[0], ret[1]};
}

std::tuple<Tensor&, Tensor&> checkpoint_nll_loss_forward_out(Tensor& a, Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      Tensor b_ = vec.at(1);
      at::nll_loss_forward_out(a_, b_, vec.at(2), vec.at(3), vec.at(4), f, g);
    };
  CheckpointTensorImpl::mutate("nll_loss_forward_out", mt, {a, b, c, d, e}, {0, 1});
  return {a, b};
}

Tensor checkpoint_nll_loss_backward(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, long e, long f, const Tensor& g) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nll_loss_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), e, f, vec.at(4))};
    };
  return CheckpointTensorImpl::make("nll_loss_backward", rt, {a, b, c, d, g})[0];
}

Tensor& checkpoint_nll_loss_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, const Tensor& e, long f, long g, const Tensor& h) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::nll_loss_backward_out(a_, vec.at(1), vec.at(2), vec.at(3), vec.at(4), f, g, vec.at(5));
    };
  CheckpointTensorImpl::mutate("nll_loss_backward_grad_input", mt, {a, b, c, d, e, h}, {0});
  return a;
}

// Tensor checkpoint_mm(const Tensor& a, const Tensor& b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::mm(vec.at(0), vec.at(1))};
//     };
//   return CheckpointTensorImpl::make("mm", rt, {a, b})[0];
// }

// Tensor& checkpoint_mm_out(Tensor& a, const Tensor& b, const Tensor& c) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0);
//       at::mm_out(a_, vec.at(1), vec.at(2));
//     };
//   CheckpointTensorImpl::mutate("mm_out", mt, {a, b, c}, {0});
//   return a;
// }

// Tensor checkpoint_sum(const Tensor& a, c10::optional<c10::ScalarType> b) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::sum(vec.at(0), b)};
//     };
//   return CheckpointTensorImpl::make("sum", rt, {a})[0];
// }

Tensor checkpoint_sum_dim_IntList(const Tensor& a, c10::ArrayRef<long> b, bool c, c10::optional<c10::ScalarType> d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("sum_dim_IntList", rt, {a})[0];
}

Tensor checkpoint_threshold(const Tensor& a, c10::Scalar b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold(vec.at(0), b, c)};
    };
  return CheckpointTensorImpl::make("threshold", rt, {a})[0];
}

Tensor& checkpoint_threshold_(Tensor& a, c10::Scalar b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_(a_, b, c);
    };
  CheckpointTensorImpl::mutate("threshold_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_threshold_out(Tensor& a, const Tensor& b, c10::Scalar c, c10::Scalar d) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::threshold_out(a_, b, c, d);
    };
  CheckpointTensorImpl::mutate("threshold_out", mt, {a}, {0});
  return a;
}

Tensor checkpoint_threshold_backward(const Tensor& a, const Tensor& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold_backward(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("threshold_backward", rt, {a, b})[0];
}

// Tensor checkpoint_select(const Tensor& a, long b, long c) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::select(vec.at(0), b, c)};
//     };
//   return CheckpointTensorImpl::make("select", rt, {a})[0];
// }

Tensor checkpoint_select_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select_backward(vec.at(0), b_, c, d)};
    };
  return CheckpointTensorImpl::make("select_backward", rt, {a})[0];
}

Tensor checkpoint_slice(const Tensor& a, long b, long c, long d, long e) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice(vec.at(0), b, c, d, e)};
    };
  return CheckpointTensorImpl::make("slice", rt, {a})[0];
}

// Tensor checkpoint_slice_backward(const Tensor& a, c10::ArrayRef<long> b, long c, long d, long e, long f) {
//   std::vector<long> b_ = b.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::slice_backward(vec.at(0), b_, c, d, e, f)};
//     };
//   return CheckpointTensorImpl::make("slice_backward", rt, {a})[0];
// }

// Tensor& checkpoint_zero_(Tensor& a) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       vec.at(0).zero_();
//     };
//   CheckpointTensorImpl::mutate("zero_", mt, {a}, {0});
//   return a;
// }

Tensor& checkpoint_squeeze_(at::Tensor& a, at::Dimname b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_();
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor& checkpoint_squeeze_(at::Tensor& a, long b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      vec.at(0).squeeze_(b);
    };
  CheckpointTensorImpl::mutate("squeeze_", mt, {a}, {0});
  return a;
}

Tensor checkpoint_sigmoid_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("sigmoid_backward", rt, {a, b})[0];
}

Tensor& checkpoint_sigmoid_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sigmoid_backward_out(a_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("sigmoid_backward_out", mt, {a, b, c}, {0});
  return a;
}

Tensor& checkpoint_sign_out(at::Tensor& a, at::Tensor const& b) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor a_ = vec.at(0);
      at::sign_out(a_, vec.at(1));
    };
  CheckpointTensorImpl::mutate("sign_out", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_sign(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sign(vec.at(0))};
    };
  return CheckpointTensorImpl::make("sign", rt, {a})[0];
}

Tensor checkpoint_tanh(const Tensor& a) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh(vec.at(0))};
    };
  return CheckpointTensorImpl::make("tanh", rt, {a})[0];
}

Tensor checkpoint_tanh_backward(at::Tensor const& a, at::Tensor const& b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tanh_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("tanh_backward", rt, {a, b})[0];
}

/// ['aten::tanh_backward_outf', 'at::Tensor &', 'tanh_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input)']
at::Tensor & checkpoint_tanh_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(2);
      return {at::tanh_backward_out(grad_input, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::tanh_backward_outf", rt, {grad_output, output, grad_input})[0];
}

// Tensor& checkpoint_tanh_backward_out(at::Tensor& a, at::Tensor const& b, at::Tensor const& c) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0);
//       at::tanh_backward_out(a_, vec.at(1), vec.at(2));
//     };
//   CheckpointTensorImpl::mutate("tanh_backward_out", mt, {a, b, c}, {0});
//   return a;
// }

// Tensor checkpoint_neg(at::Tensor const& a) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::neg(vec.at(0))};
//     };
//   return CheckpointTensorImpl::make("neg", rt, {a})[0];
// }

Tensor checkpoint_sub(at::Tensor const& a, at::Tensor const& b, c10::Scalar c) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec.at(0), vec.at(1), c)};
    };
  return CheckpointTensorImpl::make("sub", rt, {a, b})[0];
}

Tensor& checkpoint_sub_(at::Tensor& a, at::Tensor const& b, c10::Scalar c) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    self.sub_(vec.at(1), c);
  };
  CheckpointTensorImpl::mutate("sub_", mt, {a, b}, {0});
  return a;
}

// Tensor checkpoint_repeat(const at::Tensor& a, c10::ArrayRef<long> b) {
//   std::vector<long> b_ = b.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {vec.at(0).repeat(b_)};
//     };
//   return CheckpointTensorImpl::make("repeat", rt, {a})[0];
// }

// Tensor checkpoint_mean(const Tensor& self, c10::optional<c10::ScalarType> dtype) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::mean(vec[0], dtype)};
//   };
//   return CheckpointTensorImpl::make("mean", rt, {self})[0];
// }

// Tensor checkpoint_mean(const Tensor& self, IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype) {
//   std::vector<long> dim_ = dim.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::mean(vec[0], dim_, keepdim, dtype)};
//   };
//   return CheckpointTensorImpl::make("mean.dim", rt, {self})[0];
// }

Tensor checkpoint__cat(c10::ArrayRef<Tensor> a, long b) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cat(vec, b)};
    };
  std::vector<Tensor> s;
  for (const Tensor& t : a) {
    s.push_back(t);
  }
  return CheckpointTensorImpl::make("_cat", rt, s)[0];
}

Tensor& checkpoint__cat_out(Tensor& a, c10::ArrayRef<Tensor> b, long c) {
  std::vector<Tensor> args;
  args.push_back(a);
  for (const Tensor& t : b) {
    args.push_back(t);
  }
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor t = vec[0];
      at::cat_out(t, ArrayRef<Tensor>(vec.data() + 1, vec.size() - 1), c);
    };
  CheckpointTensorImpl::mutate("_cat_out", mt, args, {0});
  return a;
}

Tensor checkpoint_kl_div(at::Tensor const& a, at::Tensor const& b, long c, bool d) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::kl_div(vec.at(0), vec.at(1), c, d)};
    };
  return CheckpointTensorImpl::make("kl_div", rt, {a, b})[0];
}

// Tensor checkpoint_kl_div_backward(at::Tensor const& a, at::Tensor const& b, at::Tensor const& c, long d, bool e) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::kl_div_backward(vec.at(0), vec.at(1), vec.at(2), d, e)};
//     };
//   return CheckpointTensorImpl::make("kl_div_backward", rt, {a, b, c})[0];
// }

Tensor checkpoint_upsample_bilinear2d(at::Tensor const& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d(vec.at(0), output_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d", rt, {self})[0];
}

// Tensor& checkpoint_upsample_bilinear2d_out(at::Tensor& out, const at::Tensor& self, c10::ArrayRef<long> output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
//   std::vector<long> output_size_ = output_size.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor out = vec.at(0);
//     at::upsample_bilinear2d_out(out, vec.at(1), output_size_, align_corners, scales_h, scales_w);
//   };
//   CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {out, self}, {0});
//   return out;
// }

Tensor& checkpoint_upsample_bilinear2d_out(
    const at::Tensor& self,
    c10::ArrayRef<long> output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out) {
    
    std::vector<long> output_size_ = output_size.vec();
    mutate_function_t mt =
      [=](const Tensors& vec) {
      Tensor out = vec.at(0);
      at::upsample_bilinear2d_out(out, vec.at(1), output_size_, align_corners, scales_h, scales_w);
    };
    CheckpointTensorImpl::mutate("binary_cross_entropy_out", mt, {out, self}, {0});
    return out;
}

// Tensor& checkpoint_upsample_bilinear2d_backward_out(at::Tensor& grad_input, const at::Tensor& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
//   std::vector<long> output_size_ = output_size.vec();
//   std::vector<long> input_size_ = input_size.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor grad_input = vec.at(0);
//     at::upsample_bilinear2d_backward_out(grad_input, vec.at(1), output_size_, input_size_, align_corners, scales_h, scales_w);
//   };
//   CheckpointTensorImpl::mutate("upsample_bilinear2d_backward_out", mt, {grad_input, grad_output}, {0});
//   return grad_input;
// }

Tensor& checkpoint_upsample_bilinear2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input = vec.at(0);
    at::upsample_bilinear2d_backward_out(grad_input, vec.at(1), output_size_, input_size_, align_corners, scales_h, scales_w);
  };
  CheckpointTensorImpl::mutate("upsample_bilinear2d_backward_out", mt, {grad_input, grad_output}, {0});
  return grad_input;
}

Tensor checkpoint_upsample_bilinear2d_backward(at::Tensor const& grad_output, c10::ArrayRef<long> output_size, c10::ArrayRef<long> input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  std::vector<long> output_size_ = output_size.vec();
  std::vector<long> input_size_ = input_size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::upsample_bilinear2d_backward(vec.at(0), output_size_, input_size_, align_corners, scales_h, scales_w)};
  };
  return CheckpointTensorImpl::make("upsample_bilinear2d_backward", rt, {grad_output})[0];
}

Tensor& checkpoint_clamp_min_(Tensor& a, Scalar min) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::clamp_min_(self, min);
  };
  CheckpointTensorImpl::mutate("clamp_min_", mt, {a}, {0});
  return a;
}


/**
 * 注册接口：aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
 * 调用函数：
 * inline at::Tensor & clamp_min_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & min) {
 * return at::_ops::clamp_min_out::call(self, min, out);
 * }
*/ 
Tensor& checkpoint_clamp_min_out(const Tensor& self, const c10::Scalar& min, Tensor& out) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out = vec.at(1);
    at::clamp_min_out(out, vec.at(0), min);
  };
  CheckpointTensorImpl::mutate("clamp_min__out", mt, {self, out}, {1});
  return out;
}

Tensor checkpoint_binary_cross_entropy_with_logits(const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::binary_cross_entropy_with_logits(vec.at(0), vec.at(1), vec.at(2), vec.at(3), reduction)};
  };
  return CheckpointTensorImpl::make("binary_cross_entropy_with_logits", rt, {input, target, weight, pos_weight})[0];
}

// Tensor checkpoint_binary_cross_entropy_with_logits_backward(const Tensor& grad, const Tensor& input, const Tensor& target, const Tensor& weight, const Tensor& pos_weight, int64_t reduction) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::binary_cross_entropy_with_logits_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), reduction)};
//   };
//   return CheckpointTensorImpl::make("binary_cross_entropy_with_logits_backward", rt, {grad, input, target, weight, pos_weight})[0];
// }

std::tuple<Tensor, Tensor> checkpoint__fused_dropout(const Tensor & self, double p, c10::optional<Generator> g) {
  // TODO: Figure out how to properly duplicate the generator;
  // note that the commented-out code below results in a segfault!
  // Ref<std::shared_ptr<Generator>> gen;
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    // Generator* cur = gen.t ? gen.t.get() : g;
    // auto newG = cur->clone();
    // auto res = at::_fused_dropout(vec.at(0), p, cur);
    // gen.t = newG;
    auto res = at::_fused_dropout(vec.at(0), p, g);
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_fused_droupout_", rt, {self});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss2d_forward(at::Tensor const& self, const Tensor& target, const c10::optional<Tensor>& weight_opt, long reduction, long ignore_index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
        auto ret = at::nll_loss2d_forward(vec.at(0), vec.at(1), vec.at(2), reduction, ignore_index);
        return {std::get<0>(ret), std::get<1>(ret)};
      };
  c10::MaybeOwned<Tensor> weight_opt_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_opt_maybe_owned;
  auto ret = CheckpointTensorImpl::make("nll_loss2d_forward", rt, {self, target, weight});
  return {ret[0], ret[1]};
}

// std::tuple<Tensor&, Tensor&> checkpoint_nll_loss2d_forward_out(at::Tensor& output, at::Tensor& total_weight, const Tensor& self, const Tensor& target, const Tensor& weight, long reduction, long ignore_index) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//         Tensor out_ = vec.at(0);
//         Tensor total_weight_ = vec.at(1);
//         at::nll_loss2d_forward_out(out_, total_weight_, vec.at(2), vec.at(3), vec.at(4), reduction, ignore_index);
//       };
//   CheckpointTensorImpl::mutate("nll_loss2d_forward_out", mt, {output, total_weight, self, target, weight}, {0, 1});
//   return {output, total_weight};
// }

std::tuple<Tensor&, Tensor&> checkpoint_nll_loss2d_forward_out(const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
        Tensor out_ = vec.at(0);
        Tensor total_weight_ = vec.at(1);
        at::nll_loss2d_forward_out(out_, total_weight_, vec.at(2), vec.at(3), vec.at(4), reduction, ignore_index);
      };
  c10::MaybeOwned<Tensor> weight_opt_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_opt_maybe_owned;
  CheckpointTensorImpl::mutate("nll_loss2d_forward_out", mt, {output, total_weight, self, target, weight}, {0, 1});
  return {output, total_weight};
}

Tensor& checkpoint_nll_loss2d_backward_out(Tensor& grad_input, const Tensor& grad_output, const Tensor& self, const Tensor& target, const Tensor& weight, long reduction, long ignore_index, const Tensor& total_weight) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
        Tensor grad_input_ = vec.at(0);
        at::nll_loss2d_backward_out(grad_input_, vec.at(1), vec.at(2), vec.at(3), vec.at(4), reduction, ignore_index, vec.at(5));
      };
  CheckpointTensorImpl::mutate("nll_loss2d_backward_out", mt, {grad_input, grad_output, self, target, weight, total_weight}, {0});
  return {grad_input};
}

Tensor checkpoint_nll_loss2d_backward(const Tensor& grad_output, const Tensor& self, const Tensor& target, const c10::optional<Tensor>& weight_opt, long reduction, long ignore_index, const Tensor& total_weight) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
        return {at::nll_loss2d_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), reduction, ignore_index, vec.at(4))};
      };
  c10::MaybeOwned<Tensor> weight_opt_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_opt_maybe_owned;
  return CheckpointTensorImpl::make("nll_loss2d_backward", rt, {grad_output, self, target, weight, total_weight})[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& cx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell(vec.at(0), vec.at(1), vec.at(2),
                                         vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res), std::get<2>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell", rt,
                                        {input_gates, hidden_gates, cx, input_bias, hidden_bias});
  return {res[0], res[1], res[2]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_lstm_cell_backward(const Tensor& grad_hy, const Tensor& grad_cy, const Tensor& cx, const Tensor& cy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_lstm_cell_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_lstm_cell_backward", rt,
                                        {grad_hy, grad_cy, cx, cy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

std::tuple<Tensor, Tensor> checkpoint__thnn_fused_gru_cell(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& hx, const Tensor& input_bias, const Tensor& hidden_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4));
    return {std::get<0>(res), std::get<1>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell", rt,
                                        {input_gates, hidden_gates, hx, input_bias, hidden_bias});
  return {res[0], res[1]};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> checkpoint__thnn_fused_gru_cell_backward(const Tensor& grad_hy, const Tensor& workspace, bool has_bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto res = at::_thnn_fused_gru_cell_backward(vec.at(0), vec.at(1), has_bias);
    return {std::get<0>(res), std::get<1>(res),
        std::get<2>(res), std::get<3>(res), std::get<4>(res)};
  };
  auto res = CheckpointTensorImpl::make("_thnn_fused_gru_cell_backward", rt,
                                        {grad_hy, workspace});
  return {res[0], res[1], res[2], res[3], res[4]};
}

// Tensor& checkpoint_bitwise_and_out(Tensor& self, const Tensor& other, const Tensor& out) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor self = vec.at(0);
//     at::bitwise_and_out(self, vec.at(1), vec.at(2));
//   };
//   CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, other, out}, {0});
//   return self;
// }

Tensor& checkpoint_bitwise_and_out(Tensor& self, const Tensor& out, Scalar other) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::bitwise_and_out(self, vec.at(1), other);
  };
  CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, out}, {0});
  return self;
}

Tensor& checkpoint_fill_(Tensor& self, const Tensor& value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::fill_(self, vec.at(1));
  };
  CheckpointTensorImpl::mutate("fill_tensor", mt, {self, value}, {0});
  return self;
}

Tensor& checkpoint_fill_(Tensor& self, const at::Scalar & value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    // at::fill_(self, value);
    self.fill_(value);
  };
  CheckpointTensorImpl::mutate("fill_scalar", mt, {self}, {0});
  return self;
}

Tensor& checkpoint_masked_select_out(Tensor& self, const Tensor& mask, const Tensor& out) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::masked_select_out(self, vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("masked_select_out", mt, {self, mask, out}, {0});
  return self;
}

// Tensor checkpoint_masked_select(const Tensor& self, const Tensor& mask) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::masked_select(vec.at(0), vec.at(1))};
//   };
//   return CheckpointTensorImpl::make("masked_select", rt, {self, mask})[0];
// }

// Tensor checkpoint_index(const Tensor& self, ArrayRef<Tensor> indices) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto self = vec.at(0);
//     auto indices = std::vector<Tensor>(vec.begin() + 1, vec.end());
//     c10::List<c10::optional<Tensor>> optional_tensors;
//     for(auto& t : indices)
//       optional_tensors.push_back(t);
//     return {at::index(self, optional_tensors)};
//   };

//   std::vector<Tensor> s = {self};
//   for (const Tensor& t: indices) {
//     s.push_back(t);
//   }
//   return CheckpointTensorImpl::make("index.Tensor", rt, s)[0];
// }

// Tensor& checkpoint_index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, const bool accumulate) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor self = vec.at(0);
//     auto values = vec.at(1);
//     auto indices = std::vector<Tensor>(vec.begin() + 2, vec.end());
//     c10::List<c10::optional<Tensor>> optional_tensors;
//     for(auto& t : indices)
//       optional_tensors.push_back(t);
//     at::index_put_(self, optional_tensors, values, accumulate);
//   };
//   std::vector<Tensor> s = {self, values};
//   for (const Tensor& t: indices) {
//     s.push_back(t);
//   }
//   CheckpointTensorImpl::mutate("index_put_", mt, s, {0});
//   return self;
// }

/// ['aten::index_put_', 'at::Tensor &', 'index_put_', '(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false)']
at::Tensor & checkpoint_index_put_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      auto indices = std::vector<Tensor>(vec.begin() + 2, vec.end());
      c10::List<c10::optional<Tensor>> optional_tensors;
      for(auto& t : indices)
        optional_tensors.push_back(t);
      at::index_put_(self, optional_tensors, vec.at(1), accumulate);
    };
  std::vector<Tensor> s = {self, values};
  for(const auto& t: indices){
    c10::MaybeOwned<Tensor> t_ = at::borrow_from_optional_tensor(t);
    s.push_back((*t_));
  }
  CheckpointTensorImpl::mutate("index_put_", mt, s, {0});
  return self;
}

Tensor checkpoint_bmm(const Tensor& self, const Tensor& mat2) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::bmm(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("bmm", rt, {self, mat2})[0];
}

// Tensor checkpoint__softmax(const Tensor& self, long dim, bool half_to_float) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::_softmax(vec.at(0), dim, half_to_float)};
//   };
//   return CheckpointTensorImpl::make("_softmax", rt, {self})[0];
// }

Tensor checkpoint__softmax_backward_data(const Tensor& grad_output, const Tensor& output, int64_t dim, c10::ScalarType input_dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_softmax_backward_data(vec.at(0), vec.at(1), dim, input_dtype)};
  };
  return CheckpointTensorImpl::make("_softmax_backward_data", rt, {grad_output, output})[0];
}

// Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps
std::tuple<Tensor, Tensor, Tensor>
checkpoint_layer_norm(const Tensor& input, c10::ArrayRef<long> normalized_shape, const Tensor& weight, const Tensor& bias, float eps) {
  std::vector<long> normalized_shape_ = normalized_shape.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_layer_norm(vec.at(0), normalized_shape_, vec.at(1), vec.at(2), eps);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_layer_norm", rt, {input, weight, bias});
  return {ret[0], ret[1], ret[2]};
}

// Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask
// std::tuple<Tensor, Tensor, Tensor>
// checkpoint_layer_norm_backward(const Tensor& grad_out, const Tensor& input, c10::ArrayRef<long> normalized_shape, const Tensor& mean, const Tensor& rstd, const Tensor& weight, const Tensor& bias, std::array<bool, 3ul> output_mask) {
//   std::vector<long> normalized_shape_ = normalized_shape.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto ret = at::native_layer_norm_backward(vec.at(0), vec.at(1), normalized_shape_, vec.at(2), vec.at(3), vec.at(4), vec.at(5), output_mask);
//     return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
//   };
//   auto ret = CheckpointTensorImpl::make("native_layer_norm_backward", rt, {grad_out, input, mean, rstd, weight, bias});
//   return {ret[0], ret[1], ret[2]};
// }

/// ['aten::native_layer_norm_backward', 'std::tuple<at::Tensor,at::Tensor,at::Tensor>', 'native_layer_norm_backward', '(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask)']
std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint_native_layer_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask) {
  auto normalized_shape_ = normalized_shape.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::native_layer_norm_backward(vec.at(0), vec.at(1), normalized_shape_, vec.at(2), vec.at(3), {vec.at(4)}, {vec.at(5)}, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::native_layer_norm_backward", rt, {grad_out, input, mean, rstd, weight_, bias_});
  return {ret[0], ret[1], ret[2]};
}

// std::tuple<Tensor, Tensor>
// checkpoint_topk(const Tensor& self, long k, long dim, bool largest, bool sorted) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto ret = at::topk(vec.at(0), k, dim, largest, sorted);
//     return {std::get<0>(ret), std::get<1>(ret)};
//   };
//   auto ret = CheckpointTensorImpl::make("topk", rt, {self});
//   return {ret[0], ret[1]};
// }

std::tuple<Tensor&, Tensor&>
checkpoint_topk_values(Tensor& values, Tensor& indices, const Tensor& self, long k, long dim, bool largest, bool sorted) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor values_ = vec.at(0);
    Tensor indices_ = vec.at(1);
    at::topk_out(values_, indices_, vec.at(2), k, dim, largest, sorted);
  };
  CheckpointTensorImpl::mutate("topk_values", mt, {values, indices, self}, {0, 1});
  return {values, indices};
}

// Tensor& checkpoint_masked_fill_(Tensor& self, const Tensor& mask, const at::Scalar& value) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor self_ = vec.at(0);
//     self_.masked_fill_(vec.at(1), value);
//   };
//   CheckpointTensorImpl::mutate("masked_fill_Scalar", mt, {self, mask}, {0});
//   return {self};
// }

Tensor& checkpoint_masked_fill_(Tensor& self, const Tensor& mask, const Tensor& value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self_ = vec.at(0);
    self_.masked_fill_(vec.at(1), vec.at(2));
  };
  CheckpointTensorImpl::mutate("masked_fill_Tensor", mt, {self, mask, value}, {0});
  return {self};
}

Tensor checkpoint_clamp(const Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::clamp(vec.at(0), min, max)};
  };
  return CheckpointTensorImpl::make("clamp", rt, {self})[0];
}

Tensor& checkpoint_clamp_(Tensor& self, c10::optional<Scalar> min, c10::optional<Scalar> max) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor self_ = vec.at(0);
    at::clamp_(self_, min, max);
  };
  CheckpointTensorImpl::mutate("clamp_", mt, {self}, {0});
  return {self};
}

Tensor& checkpoint_clamp_out(const Tensor& self, const c10::optional<Scalar>& min, const c10::optional<Scalar>& max, Tensor& result) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor res = vec.at(1);
    at::clamp_outf(vec.at(0), min, max, res);
  };
  CheckpointTensorImpl::mutate("clamp_out", mt, {self, result}, {1});
  return {result};
}

// std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_thnn_conv2d_forward_out(Tensor& output, Tensor& finput, Tensor& fgrad_input, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, const Tensor& bias, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding) {
//   auto kernel_size_ = kernel_size.vec();
//   auto stride_ = stride.vec();
//   auto padding_ = padding.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor output_ = vec.at(0);
//     Tensor finput_ = vec.at(1);
//     Tensor fgrad_input_ = vec.at(2);
//     at::thnn_conv2d_forward_out(output_, finput_, fgrad_input_, vec.at(3), vec.at(4), kernel_size_, vec.at(5), stride_, padding_);
//   };
//   CheckpointTensorImpl::mutate("thnn_conv2d_forward_out", mt, {output, finput, fgrad_input, self, weight, bias}, {0, 1, 2});
//   return {output, finput, fgrad_input};
// }

// std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_forward(const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, const Tensor& bias, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding) {
//   auto kernel_size_ = kernel_size.vec();
//   auto stride_ = stride.vec();
//   auto padding_ = padding.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto ret = at::thnn_conv2d_forward(vec.at(0), vec.at(1), kernel_size_, vec.at(2), stride_, padding_);
//     return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
//   };
//   auto ret = CheckpointTensorImpl::make("thnn_conv2d_forward", rt, {self, weight, bias});
//   return {ret[0], ret[1], ret[2]};
// }

// std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_thnn_conv2d_backward_out(Tensor& grad_input, Tensor& grad_weight, Tensor& grad_bias, const Tensor& grad_output, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding, const Tensor& finput, const Tensor& fgrad_input) {
//   auto kernel_size_ = kernel_size.vec();
//   auto stride_ = stride.vec();
//   auto padding_ = padding.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor grad_input_ = vec.at(0);
//     Tensor grad_weight_ = vec.at(1);
//     Tensor grad_bias_ = vec.at(2);
//     at::thnn_conv2d_backward_out(grad_input_, grad_weight_, grad_bias_, vec.at(3), vec.at(4), vec.at(5), kernel_size_, stride_, padding_, vec.at(6), vec.at(7));
//   };
//   CheckpointTensorImpl::mutate("thnn_conv2d_backward_out", mt, {grad_input, grad_weight, grad_bias, grad_output, self, weight, finput, fgrad_input}, {0, 1, 2});
//   return {grad_input, grad_weight, grad_bias};
// }

// std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_backward(const Tensor& grad_output, const Tensor& self, const Tensor& weight, c10::ArrayRef<long> kernel_size, c10::ArrayRef<long> stride, c10::ArrayRef<long> padding, const Tensor& finput, const Tensor& fgrad_input, std::array<bool, 3ul> output_mask) {
//   auto kernel_size_ = kernel_size.vec();
//   auto stride_ = stride.vec();
//   auto padding_ = padding.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto ret = at::thnn_conv2d_backward(vec.at(0), vec.at(1), vec.at(2), kernel_size_, stride_, padding_, vec.at(3), vec.at(4), output_mask);
//     return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
//   };
//   auto ret = CheckpointTensorImpl::make("thnn_conv2d_backward", rt, {grad_output, self, weight, finput, fgrad_input});
//   return {ret[0], ret[1], ret[2]};
// }

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool training, double momentum, double eps) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::native_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), training, momentum, eps);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_batch_norm", rt, {input, weight, bias, running_mean, running_var});
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_native_batch_norm_out(Tensor& out, Tensor& save_mean, Tensor& save_invstd, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool training, double momentum, double eps) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    Tensor save_mean_ = vec.at(1);
    Tensor save_invstd_ = vec.at(2);
    at::native_batch_norm_out(out_, save_mean_, save_invstd_, vec.at(3), vec.at(4), vec.at(5), vec.at(6), vec.at(7), training, momentum, eps);
  };
  CheckpointTensorImpl::mutate("native_batch_norm_out", mt, {out, save_mean, save_invstd, input, weight, bias, running_mean, running_var}, {0, 1, 2});
  return {out, save_mean, save_invstd};
}

// std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm_backward(const Tensor& grad_out, const Tensor& input, const Tensor& weight, const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps, std::array<bool, 3ul> output_mask) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     auto ret = at::native_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), train, eps, output_mask);
//     return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
//   };
//   auto ret = CheckpointTensorImpl::make("native_batch_norm_backward", rt, {grad_out, input, weight, running_mean, running_var, save_mean, save_invstd});
//   return {ret[0], ret[1], ret[2]};
// }

std::tuple<Tensor, Tensor> checkpoint__cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, long blank, bool deterministic, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::_cudnn_ctc_loss(vec.at(0), vec.at(1), input_lengths_, target_lengths_, blank, deterministic, zero_infinity);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("_cudnn_ctc_loss", rt, {log_probs, targets});
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint__cudnn_ctc_loss_tensor(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::_cudnn_ctc_loss(vec.at(0), vec.at(1), vec.at(2), vec.at(3), blank, deterministic, zero_infinity);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("_cudnn_ctc_loss", rt, {log_probs, targets, input_lengths, target_lengths});
  return {ret[0], ret[1]};
}

std::tuple<Tensor, Tensor> checkpoint__ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths,  long blank, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    auto ret = at::_ctc_loss(vec.at(0), vec.at(1), input_lengths_, target_lengths_, blank, zero_infinity);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("_ctc_loss", rt, {log_probs, targets});
  return {ret[0], ret[1]};
}

Tensor checkpoint__ctc_loss_backward(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, const Tensor& neg_log_likelihood, const Tensor& log_alpha, long blank, bool zero_infinity) {
  auto input_lengths_ = input_lengths.vec();
  auto target_lengths_ = target_lengths.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::_ctc_loss_backward(vec.at(0), vec.at(1), vec.at(2), input_lengths_, target_lengths_, vec.at(3), vec.at(4), blank, zero_infinity)};
  };
  return CheckpointTensorImpl::make("_ctc_loss_backward", rt, {grad, log_probs, targets, neg_log_likelihood, log_alpha})[0];
}

Tensor& checkpoint_hardtanh_backward_out(Tensor& grad_input, const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor grad_input_ = vec.at(0);
    at::hardtanh_backward_out(grad_input_, vec.at(1), vec.at(2), min_val, max_val);
  };
  CheckpointTensorImpl::mutate("hardtanh_backward_out", mt, {grad_input, grad_output, self}, {0});
  return {grad_input};
}

Tensor checkpoint_hardtanh_backward(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::hardtanh_backward(vec.at(0), vec.at(1), min_val, max_val)};
  };
  return CheckpointTensorImpl::make("hardtanh_backward", rt, {grad_output, self})[0];
}

Tensor checkpoint_nonzero(const Tensor& self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::nonzero(vec.at(0))};
  };
  return CheckpointTensorImpl::make("nonzero", rt, {self})[0];
}

/// ['aten::nonzero_outf', 'at::Tensor &', 'nonzero_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_nonzero_out(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::nonzero_out(out, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::nonzero_outf", rt, {self, out})[0];
}


Tensor checkpoint_lt(const Tensor& self, Scalar other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
    return {at::lt(vec.at(0), other)};
  };
  return CheckpointTensorImpl::make("lt_Scalar", rt, {self})[0];
}

Tensor& checkpoint_lt_out(Tensor& out, const Tensor& self, Scalar other) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
    Tensor out_ = vec.at(0);
    at::lt_out(out_, vec.at(1), other);
  };
  CheckpointTensorImpl::mutate("lt_Scalar_out", mt, {out, self}, {0});
  return {out};
}

// Tensor checkpoint_lt(const Tensor& self, const Tensor& other) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::lt(vec.at(0), vec.at(1))};
//   };
//   return CheckpointTensorImpl::make("lt_Tensor", rt, {self, other})[0];
// }

// Tensor& checkpoint_lt_out(Tensor& out, const Tensor& self, const Tensor& other) {
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//     Tensor out_ = vec.at(0);
//     at::lt_out(out_, vec.at(1), vec.at(2));
//   };
//   CheckpointTensorImpl::mutate("lt_Tensor_out", mt, {out, self, other}, {0});
//   return {out};
// }

// Tensor checkpoint_any(const Tensor& self) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//     return {at::any(vec.at(0))};
//   };
//   return CheckpointTensorImpl::make("any", rt, {self})[0];
// }

bool checkpoint__use_cudnn_ctc_loss(const Tensor& log_probs, const Tensor& targets, ArrayRef<long> input_lengths, ArrayRef<long> target_lengths, long blank) {
  return at::_use_cudnn_ctc_loss(decheckpoint(log_probs), decheckpoint(targets), input_lengths, target_lengths, blank);
}

bool checkpoint_equal(const Tensor& self, const Tensor& other) {
  // there can't possibly be a reason to rematerialize
  // a single bool so we'll just compute it now
  return at::equal(decheckpoint(self), decheckpoint(other));
}

Scalar checkpoint__local_scalar_dense(at::Tensor const& a) {
  return at::_local_scalar_dense(decheckpoint(a));
}

// Tensor checkpoint_split_with_sizes_backward(c10::ArrayRef<at::Tensor> a, c10::ArrayRef<long> b, long c, c10::ArrayRef<long> d, c10::TensorOptions const& e) {
//   std::vector<Tensor> a_ = a.vec();
//   std::vector<long> d_ = d.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::split_with_sizes_backward(vec, b, c, d_, e)};
//     };
//   return CheckpointTensorImpl::make("split_with_sizes_backward", rt, a_)[0];
// }

// std::vector<Tensor> checkpoint_split_with_sizes(at::Tensor const& a, c10::ArrayRef<long> b, long c) {
//   std::vector<long> b_ = b.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return at::split_with_sizes(vec.at(0), b_, c);
//     };
//   return CheckpointTensorImpl::make("split_with_sizes", rt, {a});
// }

// Tensor checkpoint_split_backward(c10::ArrayRef<at::Tensor> a, long b, long c, c10::ArrayRef<long> d, const c10::TensorOptions& e) {
//   std::vector<Tensor> a_ = a.vec();
//   std::vector<long> d_ = d.vec();
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::split_backward(vec, b, c, d_, e)};
//     };
//   return CheckpointTensorImpl::make("split_backward", rt, a_)[0];
// }

// std::vector<Tensor> checkpoint_split(const at::Tensor& a, long b, long c) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return at::split(vec.at(0), b, c);
//     };
//   return CheckpointTensorImpl::make("split", rt, {a});
// }

Tensor checkpoint_expand(at::Tensor const& a, c10::ArrayRef<long> b, bool c) {
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).expand(b_, c)};
    };
  return CheckpointTensorImpl::make("expand", rt, {a})[0];
}

Tensor checkpoint_diag(at::Tensor const& self, long diagonal) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::diag(vec.at(0), diagonal)};
    };
  return CheckpointTensorImpl::make("diag", rt, {self})[0];
}

Tensor& checkpoint_diag_out(at::Tensor& out, const Tensor& self, long diagonal) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor out_ = vec.at(0);
      at::diag_out(out_, vec.at(1), diagonal);
    };
  CheckpointTensorImpl::mutate("diag_out", mt, {out, self}, {0});
  return {out};
}

Tensor checkpoint_mv(at::Tensor const& self, at::Tensor const& vec) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mv(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("mv", rt, {self, vec})[0];
}

Tensor& checkpoint_mv_out(at::Tensor& out, const Tensor& self, const Tensor& vec) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor out_ = vec.at(0);
      at::mv_out(out_, vec.at(1), vec.at(2));
    };
  CheckpointTensorImpl::mutate("mv_out", mt, {out, self, vec}, {0});
  return {out};
}

/////////////////////////////////// addition ///////////////////////////////////////////

/// to_copy是2.1中各种tensor.to(...)的底层函数
Tensor checkpoint__to_copy(const at::Tensor & self, at::TensorOptions options={}, bool non_blocking=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt){
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_to_copy(vec.at(0), options, non_blocking, memory_format)};
    };
  return CheckpointTensorImpl::make("_to_copy", rt, {self})[0];
}

Tensor checkpoint__to_copy(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_to_copy(vec.at(0), dtype, layout, device, pin_memory, non_blocking, memory_format)};
    };
  return CheckpointTensorImpl::make("_to_copy", rt, {self})[0];
}

template <typename T, typename = std::enable_if_t<std::is_same<T, int64_t>::value>>
at::Tensor checkpoint_view(const at::Tensor & self, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::native::view(vec.at(0), size_)};
    };
  return CheckpointTensorImpl::make("view", rt, {self})[0];
}

// template <typename T, typename = std::enable_if_t<std::is_same<T, c10::SymInt>::value>>
// at::Tensor checkpoint_view(const at::Tensor & self, c10::SymIntArrayRef size) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::native::view(vec.at(0), size)};
//     };
//   return CheckpointTensorImpl::make("view", rt, {self})[0];
// }

// at::Tensor view_dtype(const at::Tensor & self, at::ScalarType dtype);

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

at::Tensor checkpoint_addmm(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::addmm(vec.at(0), vec.at(1), vec.at(2), beta, alpha)};
      // Tensor out_ = vec.at(3);
      // return {at::addmm_outf(vec.at(0), vec.at(1), vec.at(2), beta, alpha, out_)};
    };
  // at::Tensor out = at::empty({0}).to(at::kCUDA);  // 这种行为是不安全的，相当于没有head_remat
  // out.resize_({mat1.size(0), mat2.size(1)});
  // out = out.checkpoint();
  return CheckpointTensorImpl::make("addmm", rt, {self, mat1, mat2})[0];
}

// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & checkpoint_addmm_out(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out_ = vec.at(3);
      return {at::addmm_out(out_, vec.at(0), vec.at(1), vec.at(2), beta, alpha)};
    };
  return CheckpointTensorImpl::make("addmm_out", rt, {self, mat1, mat2, out})[0];
}


std::tuple<Tensor,Tensor>
checkpoint_native_dropout(const Tensor& self, double p, c10::optional<bool> train){
  rematerialize_function_t rt =
    [p, train](const Tensors& vec) -> Tensors {
      // printf("calling at::native_dropout\n");
      auto ret = at::native_dropout(vec.at(0), p, train);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::native_dropout", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::native_dropout', 'std::tuple<at::Tensor,at::Tensor>', 'native_dropout', '(const at::Tensor & input, double p, c10::optional<bool> train)']
// std::tuple<at::Tensor,at::Tensor> checkpoint_native_dropout(const at::Tensor & input, double p, c10::optional<bool> train) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       auto ret = at::native_dropout(vec.at(0), p, train);
//       return {std::get<0>(ret), std::get<1>(ret)};
//     };
//   auto ret = CheckpointTensorImpl::make("aten::native_dropout", rt, {input});
//   return {ret[0], ret[1]};
// }


std::tuple<Tensor, Tensor, Tensor> 
checkpoint_native_batch_norm(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::native_batch_norm(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), train, momentum, epsilon);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
    c10::MaybeOwned<Tensor> weight_opt_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
    const Tensor& weight_opt_ = *weight_opt_maybe_owned;
    c10::MaybeOwned<Tensor> bias_opt_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
    const Tensor& bias_opt_ = *bias_opt_maybe_owned;
    c10::MaybeOwned<Tensor> running_mean_opt_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
    const Tensor& running_mean_opt_ = *running_mean_opt_maybe_owned;
    c10::MaybeOwned<Tensor> running_var_opt_maybe_owned = at::borrow_from_optional_tensor(running_var_opt);
    const Tensor& running_var_opt_ = *running_var_opt_maybe_owned;
  auto ret = CheckpointTensorImpl::make("native_batch_norm", rt, {self, weight_opt_, bias_opt_, running_mean_opt_, running_var_opt_});
  return {ret[0], ret[1], ret[2]};
}

/// ['std::tuple<Tensor, Tensor, Tensor, Tensor>', '_scaled_dot_product_efficient_attention_cuda', '(const Tensor& query, const Tensor& key, const Tensor& value, const c10::optional<at::Tensor>& attn_bias, bool compute_log_sumexp, double dropout_p, bool is_causal, c10::optional<double> scale)']
// std::tuple<Tensor, Tensor, Tensor, Tensor> checkpoint__scaled_dot_product_efficient_attention(
//   const Tensor& query, const Tensor& key, const Tensor& value, const c10::optional<at::Tensor>& attn_bias, 
//   bool compute_log_sumexp, double dropout_p, bool is_causal, c10::optional<double> scale) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       auto ret = at::_scaled_dot_product_efficient_attention(vec.at(0), vec.at(1), vec.at(2), vec.at(3), compute_log_sumexp, dropout_p, is_causal, scale);
//       return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
//     };
//   auto attn_bias_ = attn_bias.has_value()
//   ? c10::MaybeOwned<Tensor>::borrowed(*attn_bias)
//   : c10::MaybeOwned<Tensor>::owned(c10::in_place);
//   if(!attn_bias.has_value()){
//     int x;
//   }
//   // c10::MaybeOwned<Tensor> attn_bias_maybe_owned = at::borrow_from_optional_tensor(attn_bias);
//   // const Tensor& attn_bias_ = *attn_bias_maybe_owned;
//   auto ret = CheckpointTensorImpl::make("aten::_scaled_dot_product_efficient_attention", rt, {query, key, value, *attn_bias_});
//   return {ret[0], ret[1], ret[2], ret[3]};
// }
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> checkpoint__scaled_dot_product_efficient_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p, bool is_causal, c10::optional<double> scale) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::_scaled_dot_product_efficient_attention(vec.at(0), vec.at(1), vec.at(2), vec.at(3), compute_log_sumexp, dropout_p, is_causal, scale);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret)};
    };
  c10::MaybeOwned<Tensor> attn_bias_maybe_owned = at::borrow_from_optional_tensor(attn_bias);
  const Tensor& attn_bias_ = *attn_bias_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::_scaled_dot_product_efficient_attention", rt, {query, key, value, attn_bias_});
  return {ret[0], ret[1], ret[2], ret[3]};
}

at::Tensor checkpoint_rsqrt(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsqrt(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::rsqrt", rt, {self})[0];
}

/// ['/// ', 'at::Tensor', 'repeat', '(const at::Tensor & self, at::IntArrayRef repeats)']
at::Tensor checkpoint_repeat(const at::Tensor & self, at::IntArrayRef repeats) {
  auto repeats_ = repeats.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).repeat(repeats_)};
    };
  return CheckpointTensorImpl::make("aten::repeat", rt, {self})[0];
}

/// ['aten::_log_softmax_outf', 'at::Tensor &', '_log_softmax_outf', '(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out)']
at::Tensor & checkpoint__log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::_log_softmax_outf(vec.at(0), dim, half_to_float, out)};
    };
  return CheckpointTensorImpl::make("aten::_log_softmax_outf", rt, {self, out})[0];
}

/// ['aten::_log_softmax_backward_data_outf', 'at::Tensor &', '_log_softmax_backward_data_outf', '(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out)']
at::Tensor & checkpoint__log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::_log_softmax_backward_data_outf(vec.at(0), vec.at(1), dim, input_dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::_log_softmax_backward_data_outf", rt, {grad_output, output, out})[0];
}

/// ['aten::cross_entropy_loss_symint', 'at::Tensor', 'cross_entropy_loss_symint', '(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight={}, int64_t reduction=at::Reduction::Mean, c10::SymInt ignore_index=-100, double label_smoothing=0.0)']
at::Tensor checkpoint_cross_entropy_loss_symint(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cross_entropy_loss_symint(vec.at(0), vec.at(1), vec.at(2), reduction, ignore_index, label_smoothing)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  printf("cross_entropy_loss trigger\n");
  return CheckpointTensorImpl::make("aten::cross_entropy_loss_symint", rt, {self, target, weight_})[0];
}

/// ['aten::nll_loss_forward_outf', 'std::tuple<at::Tensor &,at::Tensor &>', 'nll_loss_forward_outf', '(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight)']
std::tuple<at::Tensor &,at::Tensor &> checkpoint_nll_loss_forward_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor output = vec.at(3);
      Tensor total_weight = vec.at(4);
      auto ret = at::nll_loss_forward_out(output, total_weight, vec.at(0), vec.at(1), vec.at(2), reduction, ignore_index);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::nll_loss_forward_out", rt, {self, target, weight_, output, total_weight});
  return {ret[0], ret[1]};
}

/// ['aten::nll_loss_backward_outf', 'at::Tensor &', 'nll_loss_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input)']
at::Tensor & checkpoint_nll_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(5);
      return {at::nll_loss_backward_outf(vec.at(0), vec.at(1), vec.at(2), vec.at(3), reduction, ignore_index, vec.at(4), grad_input)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  return CheckpointTensorImpl::make("aten::nll_loss_backward_outf", rt, {grad_output, self, target, weight_, total_weight, grad_input})[0];
}

/// ['aten::threshold_backward_outf', 'at::Tensor &', 'threshold_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input)']
at::Tensor & checkpoint_threshold_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(2);
      return {at::threshold_backward_outf(vec.at(0), vec.at(1), threshold, grad_input)};
    };
  return CheckpointTensorImpl::make("aten::threshold_backward_outf", rt, {grad_output, self, grad_input})[0];
}

/// ['aten::threshold_backward', 'at::Tensor', 'threshold_backward', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold)']
at::Tensor checkpoint_threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::threshold_backward(vec.at(0), vec.at(1), threshold)};
    };
  return CheckpointTensorImpl::make("aten::threshold_backward", rt, {grad_output, self})[0];
}

/// ['aten::silu_backward', 'at::Tensor', 'silu_backward', '(const at::Tensor & grad_output, const at::Tensor & self)']
at::Tensor checkpoint_silu_backward(const at::Tensor & grad_output, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::silu_backward(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::silu_backward", rt, {grad_output, self})[0];
}

/// ['aten::silu_backward_outf', 'at::Tensor &', 'silu_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input)']
at::Tensor & checkpoint_silu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(2);
      return {at::silu_backward_outf(vec.at(0), vec.at(1), grad_input)};
    };
  return CheckpointTensorImpl::make("aten::silu_backward_outf", rt, {grad_output, self, grad_input})[0];
}

/// ['aten::sigmoid_outf', 'at::Tensor &', 'sigmoid_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_sigmoid_out(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::sigmoid_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::sigmoid_outf", rt, {self, out})[0];
}

/// ['aten::sigmoid', 'at::Tensor', 'sigmoid', '(const at::Tensor & self)']
at::Tensor checkpoint_sigmoid(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sigmoid(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::sigmoid", rt, {self})[0];
}

/// ['aten::sub_', 'at::Tensor &', 'sub_', '(Tensor& self, const Tensor& other, const Scalar& alpha)']
at::Tensor & checkpoint_sub_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      self.sub_(vec.at(1), alpha);
      // return {at::sub_(self, vec.at(1), alpha)};
    };
  CheckpointTensorImpl::mutate("aten::sub_tensor", mt, {self, other}, {0});
  return self;
}

/// ['aten::sub_', 'at::Tensor &', 'sub_', '(Tensor& self, const Tensor& other, const Scalar& alpha)']
at::Tensor & checkpoint_sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      self.sub_(other, alpha);
      // return {at::sub_(self, vec.at(1), alpha)};
    };
  CheckpointTensorImpl::mutate("aten::sub_scalar", mt, {self}, {0});
  return self;
}

/// ['aten::slice_backward', 'at::Tensor', 'slice_backward', '(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step)']
at::Tensor checkpoint_slice_backward(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto input_sizes_ = input_sizes.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice_backward(vec.at(0), input_sizes_, dim, start, end, step)};
    };
  return CheckpointTensorImpl::make("aten::slice_backward", rt, {grad_output})[0];
}

/// ['aten::native_dropout_backward', 'at::Tensor', 'native_dropout_backward', '(const at::Tensor & grad_output, const at::Tensor & mask, double scale)']
at::Tensor checkpoint_native_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double scale) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::native_dropout_backward(vec.at(0), vec.at(1), scale)};
    };
  return CheckpointTensorImpl::make("aten::native_dropout_backward", rt, {grad_output, mask})[0];
}

/// ['aten::_foreach_norm', 'std::vector<at::Tensor>', '_foreach_norm', '(at::TensorList self, const at::Scalar & ord=2)']
std::vector<at::Tensor> checkpoint__foreach_norm(at::TensorList self, const at::Scalar & ord) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_foreach_norm(at::TensorList(vec), ord)};
    };
  Tensors inputs;
  for (const auto i : c10::irange(self.size())){
    inputs.push_back(self[i]);
  }
  return CheckpointTensorImpl::make("aten::_foreach_norm", rt, {inputs});
}

/// ['aten::linalg_vector_norm_outf', 'at::Tensor &', 'linalg_vector_norm_outf', '(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_linalg_vector_norm_out(const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      if(dim.has_value())
        return {at::linalg_vector_norm_outf(vec.at(0), ord, dim_, keepdim, dtype, out)};
      else
        return {at::linalg_vector_norm_outf(vec.at(0), ord, dim, keepdim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::linalg_vector_norm_outf", rt, {self, out})[0];
}

/// ['aten::_foreach_mul_', 'void', '_foreach_mul_', '(at::TensorList self, const at::Tensor & other)']
void checkpoint__foreach_mul_(at::TensorList self, const at::Tensor & other) {
  Tensors inputs;
  for (const auto i : c10::irange(self.size())){
    inputs.push_back(self[i].decheckpoint());
  }
  at::_foreach_mul_(at::TensorList(inputs), other.decheckpoint());
}

/// ['aten::_foreach_mul_.Scalar', 'void', '_foreach_mul_', '(at::TensorList self, const at::Tensor & other)']
void checkpoint__foreach_mul_(at::TensorList self, const at::Scalar & scalar) {
  Tensors inputs;
  for (const auto i : c10::irange(self.size())){
    inputs.push_back(self[i].decheckpoint());
  }
  at::_foreach_mul_(at::TensorList(inputs), scalar);
}

/// ['aten::embedding_dense_backward', 'at::Tensor', 'embedding_dense_backward', '(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq)']
at::Tensor checkpoint_embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_dense_backward(vec.at(0), vec.at(1), num_weights, padding_idx, scale_grad_by_freq)};
    };
  return CheckpointTensorImpl::make("aten::embedding_dense_backward", rt, {grad_output, indices})[0];
}

/// ['aten::zero_', 'at::Tensor &', 'zero_', '(at::Tensor & self)']
at::Tensor & checkpoint_zero_(at::Tensor & self) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      at::zero_(self);
    };
  CheckpointTensorImpl::mutate("zero_", mt, {self}, {0});
  return {self};
}

/// ['aten::_slow_conv2d_forward_out', 'at::Tensor &', '_slow_conv2d_forward_out', '(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding)']
at::Tensor & checkpoint__slow_conv2d_forward_out(at::Tensor & output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor output = vec.at(0);
      return {at::_slow_conv2d_forward_out(output, vec.at(1), vec.at(2), kernel_size_, vec.at(3), stride_, padding_)};
    };
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::_slow_conv2d_forward_out", rt, {output, self, weight, bias_})[0];
}


/// ['aten::_slow_conv2d_forward_outf', 'at::Tensor &', '_slow_conv2d_forward_outf', '(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output)']
at::Tensor & checkpoint__slow_conv2d_forward_outf(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor output = vec.at(3);
      return {at::_slow_conv2d_forward_outf(vec.at(0), vec.at(1), kernel_size_, vec.at(2), stride_, padding_, output)};
    };
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::_slow_conv2d_forward_outf", rt, {self, weight, bias_, output})[0];
}

/// ['aten::_slow_conv2d_forward', 'at::Tensor', '_slow_conv2d_forward', '(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding)']
at::Tensor checkpoint__slow_conv2d_forward(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_slow_conv2d_forward(vec.at(0), vec.at(1), kernel_size_, vec.at(2), stride_, padding_)};
    };
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::_slow_conv2d_forward", rt, {self, weight, bias_})[0];
}

/// ['aten::_slow_conv2d_backward.grad_input', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', '_slow_conv2d_backward_out', '(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint__slow_conv2d_backward_out(at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(0);
      Tensor grad_weight = vec.at(1);
      Tensor grad_bias = vec.at(2);
      auto ret = at::_slow_conv2d_backward_out(grad_input, grad_weight, grad_bias, vec.at(3), vec.at(4), vec.at(5), kernel_size_, stride_, padding_);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_slow_conv2d_backward.grad_input", rt, {grad_input, grad_weight, grad_bias, grad_output, self, weight});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::_slow_conv2d_backward.grad_input', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', '_slow_conv2d_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint__slow_conv2d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(3);
      Tensor grad_weight = vec.at(4);
      Tensor grad_bias = vec.at(5);
      auto ret = at::_slow_conv2d_backward_outf(vec.at(0), vec.at(1), vec.at(2), kernel_size_, stride_, padding_, grad_input, grad_weight, grad_bias);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_slow_conv2d_backward.grad_input", rt, {grad_output, self, weight, grad_input, grad_weight, grad_bias});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::_slow_conv2d_backward.output_mask', '::std::tuple<at::Tensor,at::Tensor,at::Tensor>', '_slow_conv2d_backward', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask)']
::std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint__slow_conv2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::_slow_conv2d_backward(vec.at(0), vec.at(1), vec.at(2), kernel_size_, stride_, padding_, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_slow_conv2d_backward.output_mask", rt, {grad_output, self, weight});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::_slow_conv2d_backward.output_mask_out', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', '_slow_conv2d_backward_out', '(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint__slow_conv2d_backward_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out0 = vec.at(0);
      Tensor out1 = vec.at(1);
      Tensor out2 = vec.at(2);
      auto ret = at::_slow_conv2d_backward_out(out0, out1, out2, vec.at(3), vec.at(4), vec.at(5), kernel_size_, stride_, padding_, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_slow_conv2d_backward.output_mask_out", rt, {out0, out1, out2, grad_output, self, weight});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::_slow_conv2d_backward.output_mask_out', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', '_slow_conv2d_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint__slow_conv2d_backward_outf(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out0 = vec.at(3);
      Tensor out1 = vec.at(4);
      Tensor out2 = vec.at(5);
      auto ret = at::_slow_conv2d_backward_outf(vec.at(0), vec.at(1), vec.at(2), kernel_size_, stride_, padding_, output_mask, out0, out1, out2);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_slow_conv2d_backward.output_mask_out", rt, {grad_output, self, weight, out0, out1, out2});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::max_pool2d_with_indices.out', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_pool2d_with_indices_out', '(at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride={}, at::IntArrayRef padding=0, at::IntArrayRef dilation=1, bool ceil_mode=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_pool2d_with_indices_out(at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::max_pool2d_with_indices_out(out, indices, vec.at(2), kernel_size_, stride_, padding_, dilation_, ceil_mode);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max_pool2d_with_indices.out", rt, {out, indices, self});
  return {ret[0], ret[1]};
}

/// TODO: BUG max_pool2d_with_indices produces leaky memory for some reason unknown(temporarily)
/// ['aten::max_pool2d_with_indices.out', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_pool2d_with_indices_outf', '(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_pool2d_with_indices_outf(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::max_pool2d_with_indices_outf(vec.at(0), kernel_size_, stride_, padding_, dilation_, ceil_mode, out, indices);
      // at::max_pool2d_with_indices_out(out, indices, vec.at(0), kernel_size_, stride_, padding_, dilation_, ceil_mode);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max_pool2d_with_indices.out", rt, {self, out, indices});
  return {ret[0], ret[1]};
  // out = out.checkpoint();
  // indices = indices.checkpoint();
  // CheckpointTensorImpl::mutate("aten::max_pool2d_with_indices.out", mt, {self, out, indices}, {1, 2});
  // return {out, indices};
}

/// ['aten::max_pool2d_with_indices', '::std::tuple<at::Tensor,at::Tensor>', 'max_pool2d_with_indices', '(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride={}, at::IntArrayRef padding=0, at::IntArrayRef dilation=1, bool ceil_mode=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_max_pool2d_with_indices(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max_pool2d_with_indices(vec.at(0), kernel_size_, stride_, padding_, dilation_, ceil_mode);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max_pool2d_with_indices", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::native_batch_norm_backward', '::std::tuple<at::Tensor,at::Tensor,at::Tensor>', 'native_batch_norm_backward', '(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask)']
::std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint_native_batch_norm_backward(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::native_batch_norm_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), train, eps, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean);
  const Tensor& running_mean_ = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var);
  const Tensor& running_var_ = *running_var_maybe_owned;
  c10::MaybeOwned<Tensor> save_mean_maybe_owned = at::borrow_from_optional_tensor(save_mean);
  const Tensor& save_mean_ = *save_mean_maybe_owned;
  c10::MaybeOwned<Tensor> save_invstd_maybe_owned = at::borrow_from_optional_tensor(save_invstd);
  const Tensor& save_invstd_ = *save_invstd_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::native_batch_norm_backward", rt, {grad_out, input, weight_, running_mean_, running_var_, save_mean_, save_invstd_});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::native_batch_norm_backward.out', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', 'native_batch_norm_backward_out', '(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint_native_batch_norm_backward_out(at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out0 = vec.at(0);
      Tensor out1 = vec.at(1);
      Tensor out2 = vec.at(2);
      auto ret = at::native_batch_norm_backward_out(out0, out1, out2, vec.at(3), vec.at(4), vec.at(5), vec.at(6), vec.at(7), vec.at(8), vec.at(9), train, eps, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean);
  const Tensor& running_mean_ = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var);
  const Tensor& running_var_ = *running_var_maybe_owned;
  c10::MaybeOwned<Tensor> save_mean_maybe_owned = at::borrow_from_optional_tensor(save_mean);
  const Tensor& save_mean_ = *save_mean_maybe_owned;
  c10::MaybeOwned<Tensor> save_invstd_maybe_owned = at::borrow_from_optional_tensor(save_invstd);
  const Tensor& save_invstd_ = *save_invstd_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::native_batch_norm_backward.out", rt, {out0, out1, out2, grad_out, input, weight_, running_mean_, running_var_, save_mean_, save_invstd_});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::native_batch_norm_backward.out', '::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &>', 'native_batch_norm_backward_outf', '(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2)']
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> checkpoint_native_batch_norm_backward_outf(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out0 = vec.at(7);
      Tensor out1 = vec.at(8);
      Tensor out2 = vec.at(9);
      auto ret = at::native_batch_norm_backward_outf(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), train, eps, output_mask, out0, out1, out2);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean);
  const Tensor& running_mean_ = *running_mean_maybe_owned;
  c10::MaybeOwned<Tensor> running_var_maybe_owned = at::borrow_from_optional_tensor(running_var);
  const Tensor& running_var_ = *running_var_maybe_owned;
  c10::MaybeOwned<Tensor> save_mean_maybe_owned = at::borrow_from_optional_tensor(save_mean);
  const Tensor& save_mean_ = *save_mean_maybe_owned;
  c10::MaybeOwned<Tensor> save_invstd_maybe_owned = at::borrow_from_optional_tensor(save_invstd);
  const Tensor& save_invstd_ = *save_invstd_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::native_batch_norm_backward.out", rt, {grad_out, input, weight_, running_mean_, running_var_, save_mean_, save_invstd_, out0, out1, out2});
  return {ret[0], ret[1], ret[2]};
}

// Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(Tensor& a, const Tensor& b, const Tensor& c, c10::ArrayRef<long> d, c10::ArrayRef<long> e, c10::ArrayRef<long> f, c10::ArrayRef<long> g, bool h, const Tensor& i) {
//   std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor a_ = vec.at(0);
//       at::max_pool2d_with_indices_backward_out(a_, vec.at(1), vec.at(2), d_, e_, f_, g, h, vec.at(3));
//     };
//   CheckpointTensorImpl::mutate("max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
//   return a;
// }

/// ['aten::max_pool2d_with_indices_backward.grad_input', 'at::Tensor &', 'max_pool2d_with_indices_backward_out', '(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices)']
// at::Tensor & checkpoint_max_pool2d_with_indices_backward_out(at::Tensor & grad_input, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
//   auto kernel_size_ = kernel_size.vec();
//   auto stride_ = stride.vec();
//   auto padding_ = padding.vec();
//   auto dilation_ = dilation.vec();
//   mutate_function_t mt =
//     [=](const Tensors& vec) {
//       Tensor grad_input = vec.at(0);
//       at::max_pool2d_with_indices_backward_out(grad_input, vec.at(1), vec.at(2), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(3));
//     };
//   CheckpointTensorImpl::mutate("aten::max_pool2d_with_indices_backward.grad_input", mt, {grad_input, grad_output, self, indices}, {0});
//   return grad_input;
// }

/// TODO: BUG max_pool2d_with_indices_backward_out produces leaky memory for some reason unknown(temporarily) 暂时通过这样的方式解决了，并没有解决本质问题
/// ['aten::max_pool2d_with_indices_backward.grad_input', 'at::Tensor &', 'max_pool2d_with_indices_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input)']
at::Tensor & checkpoint_max_pool2d_with_indices_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  // rematerialize_function_t rt =
  //   [=](const Tensors& vec) -> Tensors {
  //     Tensor grad_input = vec.at(3);
  //     // at::max_pool2d_with_indices_backward_out(grad_input, vec.at(0), vec.at(1), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(2))
  //     return {at::max_pool2d_with_indices_backward_outf(vec.at(0), vec.at(1), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(2), grad_input)};
  //   };
  // return CheckpointTensorImpl::make("aten::max_pool2d_with_indices_backward.grad_input", rt, {grad_output, self, indices, grad_input})[0];
  mutate_function_t mt =
  [=](const Tensors& vec) {
    Tensor grad_input = vec.at(0);
    at::max_pool2d_with_indices_backward_out(grad_input, vec.at(1), vec.at(2), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(3));
  };
  grad_input = grad_input.checkpoint();
  CheckpointTensorImpl::mutate("aten::max_pool2d_with_indices_backward.grad_input", mt, {grad_input, grad_output, self, indices}, {0});
  return grad_input;
}

/// ['aten::max_pool2d_with_indices_backward', 'at::Tensor', 'max_pool2d_with_indices_backward', '(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices)']
at::Tensor checkpoint_max_pool2d_with_indices_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max_pool2d_with_indices_backward(vec.at(0), vec.at(1), kernel_size_, stride_, padding_, dilation_, ceil_mode, vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::max_pool2d_with_indices_backward", rt, {grad_output, self, indices})[0];
}

/// ['aten::_index_put_impl_', 'at::Tensor &', '_index_put_impl_', '(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate=false, bool unsafe=false)']
at::Tensor & checkpoint__index_put_impl_(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      c10::List<c10::optional<at::Tensor>> inds;
      for(int i=2; i<vec.size(); i++){
        c10::optional<at::Tensor> t_(vec.at(i));
        inds.push_back(t_);
      }
      at::_index_put_impl_(self, inds, vec.at(1), accumulate, unsafe);
    };
  std::vector<Tensor> inputs = {self, values};
  for (auto i : c10::irange(indices.size())) {
    auto t = indices.get(i);
    c10::MaybeOwned<Tensor> t_maybe_owned = at::borrow_from_optional_tensor(t);
    const Tensor& t_ = *t_maybe_owned;
    inputs.push_back(t_);
  }
  CheckpointTensorImpl::mutate("_index_put_impl_", mt, inputs, {0});
  return self;
}

/// ['aten::_foreach_lerp_', 'void', '_foreach_lerp_', '(at::TensorList self, at::TensorList tensors1, const at::Scalar & weight)']
void checkpoint__foreach_lerp_(at::TensorList self, at::TensorList tensors1, const at::Scalar & weight) {
  Tensors self_, tensors1_;
  for (const auto i : c10::irange(self.size())){
    self_.push_back(self[i].decheckpoint());
  }
  for (const auto i : c10::irange(tensors1.size())){
    tensors1_.push_back(tensors1[i].decheckpoint());
  }
  at::_foreach_lerp_(at::TensorList(self_), at::TensorList(tensors1_), weight);
}

/// ['aten::_foreach_addcmul_', 'void', '_foreach_addcmul_', '(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value=1)']
void checkpoint__foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  Tensors self_;
  for (const auto i : c10::irange(self.size())) {
    self_.push_back(self[i].decheckpoint());
  }
  Tensors tensor1_;
  for (const auto i : c10::irange(tensor1.size())) {
    tensor1_.push_back(tensor1[i].decheckpoint());
  }
  Tensors tensor2_;
  for (const auto i : c10::irange(tensor2.size())) {
    tensor2_.push_back(tensor2[i].decheckpoint());
  }
  at::_foreach_addcmul_(at::TensorList(self_), at::TensorList(tensor1_), at::TensorList(tensor2_), value);
}

/// ['aten::native_layer_norm', 'std::tuple<at::Tensor,at::Tensor,at::Tensor>', 'native_layer_norm', '(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps)']
std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint_native_layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
  auto normalized_shape_ = normalized_shape.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::native_layer_norm(vec.at(0), normalized_shape_, vec.at(1), vec.at(2), eps);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight);
  const Tensor& weight_ = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
  const Tensor& bias_ = *bias_maybe_owned;
  auto ret = CheckpointTensorImpl::make("aten::native_layer_norm", rt, {input, weight_, bias_});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::where.self', 'at::Tensor', 'where', '(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_where(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::where(vec.at(0), vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::where", rt, {condition, self, other})[0];
}

/// ['aten::exp_outf', 'at::Tensor &', 'exp_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_exp_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::exp_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::exp_outf", rt, {self, out})[0];
}

/// ['aten::log_outf', 'at::Tensor &', 'log_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_log_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::log_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::log_outf", rt, {self, out})[0];
}

/// ['aten::_foreach_sqrt', 'std::vector<at::Tensor>', '_foreach_sqrt', '(at::TensorList self)']
std::vector<at::Tensor> checkpoint__foreach_sqrt(at::TensorList self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_foreach_sqrt(at::TensorList(vec))};
    };
  Tensors inputs;
  for (const auto i : c10::irange(self.size())){
    inputs.push_back(self[i]);
  }
  return CheckpointTensorImpl::make("aten::_foreach_sqrt", rt, {inputs});
}

/// ['aten::_foreach_div_', 'void', '_foreach_div_', '(at::TensorList self, at::ArrayRef<at::Scalar> scalars)']
void checkpoint__foreach_div_(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  Tensors self_;
  for (const auto i : c10::irange(self.size())) {
    self_.push_back(self[i].decheckpoint());
  }
  at::_foreach_div_(at::TensorList(self_), scalars);
}

/// ['aten::_foreach_add_', 'void', '_foreach_add_', '(at::TensorList self, const at::Scalar & scalar)']
void checkpoint__foreach_add_(at::TensorList self, const at::Scalar & scalar) {
  Tensors self_;
  for (const auto i : c10::irange(self.size())) {
    self_.push_back(self[i].decheckpoint());
  }
  at::_foreach_add_(at::TensorList(self_), scalar);
}

/// ['aten::_foreach_addcdiv_', 'void', '_foreach_addcdiv_', '(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)']
void checkpoint__foreach_addcdiv_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  Tensors self_;
  for (const auto i : c10::irange(self.size())) {
    self_.push_back(self[i].decheckpoint());
  }
  Tensors tensor1_;
  for (const auto i : c10::irange(tensor1.size())) {
    tensor1_.push_back(tensor1[i].decheckpoint());
  }
  Tensors tensor2_;
  for (const auto i : c10::irange(tensor2.size())) {
    tensor2_.push_back(tensor2[i].decheckpoint());
  }
  at::_foreach_addcdiv_(at::TensorList(self_), at::TensorList(tensor1_), at::TensorList(tensor2_), scalars);
}

/// ['aten::baddbmm_outf', 'at::Tensor &', 'baddbmm_outf', '(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_baddbmm_outf(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(3);
      return {at::baddbmm_out(out, vec.at(0), vec.at(1), vec.at(2), beta, alpha)};
    };
  return CheckpointTensorImpl::make("aten::baddbmm_outf", rt, {self, batch1, batch2, out})[0];
}

/// ['aten::masked_fill_', 'at::Tensor &', 'masked_fill_', '(Tensor& self, const Tensor & mask, const Scalar& value)']
at::Tensor & checkpoint_masked_fill_(Tensor& self, const Tensor & mask, const Scalar& value) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      self.masked_fill_(vec.at(1), value);
    };
  CheckpointTensorImpl::mutate("masked_fill_", mt, {self, mask}, {0});
  return {self};
}

/// ['aten::logical_or', 'at::Tensor', 'logical_or', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_logical_or(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::logical_or(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::logical_or", rt, {self, other})[0];
}

/// ['aten::logical_or_outf', 'at::Tensor &', 'logical_or_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_logical_or_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::logical_or_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::logical_or_outf", rt, {self, other, out})[0];
}

/// ['aten::bitwise_or_outf', 'at::Tensor &', 'bitwise_or_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_bitwise_or_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::bitwise_or_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_or_outf", rt, {self, other, out})[0];
}

/// ['aten::bitwise_or', 'at::Tensor', 'bitwise_or', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_bitwise_or(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::bitwise_or(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::bitwise_or", rt, {self, other})[0];
}

/// ['aten::masked_select', 'at::Tensor', 'masked_select', '(const at::Tensor & self, const at::Tensor & mask)']
at::Tensor checkpoint_masked_select(const at::Tensor & self, const at::Tensor & mask) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::masked_select(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::masked_select", rt, {self, mask})[0];
}

/// ['aten::ceil_outf', 'at::Tensor &', 'ceil_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_ceil_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::ceil_out(out, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::ceil_outf", rt, {self, out})[0];
}

/// ['aten::_amp_foreach_non_finite_check_and_unscale_', 'void', '_amp_foreach_non_finite_check_and_unscale_', '(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale)']
void checkpoint__amp_foreach_non_finite_check_and_unscale_(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
  Tensors self_;
  for (const auto i : c10::irange(self.size())) {
    self_.push_back(self[i].decheckpoint());
  }
  at::_amp_foreach_non_finite_check_and_unscale_(at::TensorList(self_), found_inf, inv_scale.decheckpoint());
}

/// ['aten::avg_pool2d_outf', 'at::Tensor &', 'avg_pool2d_outf', '(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out)']
at::Tensor & checkpoint_avg_pool2d_out(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::avg_pool2d_out(out, vec.at(0), kernel_size_, stride_, padding_, ceil_mode, count_include_pad, divisor_override)};
    };
  return CheckpointTensorImpl::make("aten::avg_pool2d_outf", rt, {self, out})[0];
}

/// ['aten::avg_pool2d_backward_outf', 'at::Tensor &', 'avg_pool2d_backward_outf', '(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input)']
at::Tensor & checkpoint_avg_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  auto kernel_size_ = kernel_size.vec();
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor grad_input = vec.at(2);
      return {at::avg_pool2d_backward_out(grad_input, vec.at(0), vec.at(1), kernel_size_, stride_, padding_, ceil_mode, count_include_pad, divisor_override)};
    };
  return CheckpointTensorImpl::make("aten::avg_pool2d_backward_outf", rt, {grad_output, self, grad_input})[0];
}

/// ['aten::convolution_backward', 'std::tuple<at::Tensor,at::Tensor,at::Tensor>', 'convolution_backward', '(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask)']
std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint_convolution_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalIntArrayRef bias_sizes, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
  c10::IntArrayRef bias_sizes_;
  if(bias_sizes.has_value()){
    bias_sizes_ = bias_sizes.value();
  }
  auto stride_ = stride.vec();
  auto padding_ = padding.vec();
  auto dilation_ = dilation.vec();
  auto output_padding_ = output_padding.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::convolution_backward(vec.at(0), vec.at(1), vec.at(2), {bias_sizes_}, stride_, padding_, dilation_, transposed, output_padding_, groups, output_mask);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::convolution_backward", rt, {grad_output, input, weight});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::new_zeros', 'at::Tensor', 'new_zeros', '(const at::Tensor & self, c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_new_zeros(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::native::new_zeros(vec.at(0), size_, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::new_zeros", rt, {self})[0];
}

/// ['aten::alias', 'at::Tensor', 'alias', '(const at::Tensor & self)']
at::Tensor checkpoint_alias(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::alias(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::alias", rt, {self})[0];
}

/// ['aten::norm', 'at::Tensor', 'norm', '(const at::Tensor & self, const at::Scalar & p=2)']
at::Tensor checkpoint_norm(const at::Tensor & self, const at::Scalar & p) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::norm(vec.at(0), p)};
    };
  return CheckpointTensorImpl::make("aten::norm", rt, {self})[0];
}

/// ['aten::norm', 'at::Tensor', 'norm', '(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype)']
at::Tensor checkpoint_norm(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  auto dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::norm(vec.at(0), p, dim_, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::norm", rt, {self})[0];
}

void checkpoint_record_stream(Tensor& self, c10::Stream stream){
  self.decheckpoint().record_stream(stream);
}

/// ['aten::std', 'at::Tensor', 'std', '(const at::Tensor & self, at::OptionalIntArrayRef dim=c10::nullopt, const c10::optional<at::Scalar> & correction=c10::nullopt, bool keepdim=false)']
at::Tensor checkpoint_std(const at::Tensor & self, at::OptionalIntArrayRef dim, const c10::optional<at::Scalar> & correction, bool keepdim) {
  c10::IntArrayRef dim_;
  if(dim.has_value()){
    dim_ = dim.value();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::std(vec.at(0), {dim_}, correction, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::std", rt, {self})[0];
}

/// ['aten::softplus', 'at::Tensor', 'softplus', '(const at::Tensor & self, const at::Scalar & beta=1, const at::Scalar & threshold=20)']
at::Tensor checkpoint_softplus(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::softplus(vec.at(0), beta, threshold)};
    };
  return CheckpointTensorImpl::make("aten::softplus", rt, {self})[0];
}

/// ['aten::softplus_backward', 'at::Tensor', 'softplus_backward', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold)']
at::Tensor checkpoint_softplus_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::softplus_backward(vec.at(0), vec.at(1), beta, threshold)};
    };
  return CheckpointTensorImpl::make("aten::softplus_backward", rt, {grad_output, self})[0];
}

/// ['aten::pad', 'at::Tensor', 'pad', '(const at::Tensor & self, at::IntArrayRef pad, c10::string_view mode="constant", c10::optional<double> value=c10::nullopt)']
at::Tensor checkpoint_pad(const at::Tensor & self, at::IntArrayRef pad, c10::string_view mode, c10::optional<double> value) {
  auto pad_ = pad.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::pad(vec.at(0), pad_, mode, value)};
    };
  return CheckpointTensorImpl::make("aten::pad", rt, {self})[0];
}

/// ['aten::maximum', 'at::Tensor', 'maximum', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_maximum(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::maximum(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::maximum", rt, {self, other})[0];
}

/// ['aten::amax', 'at::Tensor', 'amax', '(const at::Tensor & self, at::IntArrayRef dim={}, bool keepdim=false)']
at::Tensor checkpoint_amax(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  auto dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::amax(vec.at(0), dim_, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::amax", rt, {self})[0];
}

/// ['aten::log_sigmoid_forward', 'std::tuple<at::Tensor,at::Tensor>', 'log_sigmoid_forward', '(const at::Tensor & self)']
std::tuple<at::Tensor,at::Tensor> checkpoint_log_sigmoid_forward(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::log_sigmoid_forward(vec.at(0));
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::log_sigmoid_forward", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::log_sigmoid_backward', 'at::Tensor', 'log_sigmoid_backward', '(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer)']
at::Tensor checkpoint_log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::log_sigmoid_backward(vec.at(0), vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::log_sigmoid_backward", rt, {grad_output, self, buffer})[0];
}

/// ['aten::logical_not_outf', 'at::Tensor &', 'logical_not_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_logical_not_out(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::logical_not_out(out, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::logical_not_outf", rt, {self, out})[0];
}

/// ['aten::nansum', 'at::Tensor', 'nansum', '(const at::Tensor & self, at::OptionalIntArrayRef dim=c10::nullopt, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_nansum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  c10::IntArrayRef dim_;
  if(dim.has_value()){
    dim_ = dim.value();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nansum(vec.at(0), {dim_}, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::nansum", rt, {self})[0];
}

/// ['aten::nan_to_num_outf', 'at::Tensor &', 'nan_to_num_outf', '(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out)']
at::Tensor & checkpoint_nan_to_num_out(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::nan_to_num_out(out, vec.at(0), nan, posinf, neginf)};
    };
  return CheckpointTensorImpl::make("aten::nan_to_num_outf", rt, {self, out})[0];
}

/// ['aten::nan_to_num', 'at::Tensor', 'nan_to_num', '(const at::Tensor & self, c10::optional<double> nan=c10::nullopt, c10::optional<double> posinf=c10::nullopt, c10::optional<double> neginf=c10::nullopt)']
at::Tensor checkpoint_nan_to_num(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nan_to_num(vec.at(0), nan, posinf, neginf)};
    };
  return CheckpointTensorImpl::make("aten::nan_to_num", rt, {self})[0];
}

/// ['aten::nanmean', 'at::Tensor', 'nanmean', '(const at::Tensor & self, at::OptionalIntArrayRef dim=c10::nullopt, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_nanmean(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  c10::IntArrayRef dim_;
  if(dim.has_value()){
    dim_ = dim.value();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::nanmean(vec.at(0), {dim_}, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::nanmean", rt, {self})[0];
}

/// ['aten::floor', 'at::Tensor', 'floor', '(const at::Tensor & self)']
at::Tensor checkpoint_floor(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::floor(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::floor", rt, {self})[0];
}

/// ['aten::minimum', 'at::Tensor', 'minimum', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_minimum(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::minimum(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::minimum", rt, {self, other})[0];
}

/// ['aten::sgn', 'at::Tensor', 'sgn', '(const at::Tensor & self)']
at::Tensor checkpoint_sgn(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sgn(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::sgn", rt, {self})[0];
}

/// ['aten::logical_and_outf', 'at::Tensor &', 'logical_and_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_logical_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::logical_and_out(out, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::logical_and_outf", rt, {self, other, out})[0];
}

/// ['aten::argmin_outf', 'at::Tensor &', 'argmin_outf', '(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out)']
at::Tensor & checkpoint_argmin_out(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::argmin_out(out, vec.at(0), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::argmin_outf", rt, {self, out})[0];
}

/// ['aten::gelu_backward', 'at::Tensor', 'gelu_backward', '(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate="none")']
at::Tensor checkpoint_gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::gelu_backward(vec.at(0), vec.at(1), approximate)};
    };
  return CheckpointTensorImpl::make("aten::gelu_backward", rt, {grad_output, self})[0];
}


/// ['aten::gelu_outf', 'at::Tensor &', 'gelu_outf', '(const at::Tensor & self, c10::string_view approximate, at::Tensor & out)']
at::Tensor & checkpoint_gelu_out(const at::Tensor & self, c10::string_view approximate, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::gelu_out(out, vec.at(0), approximate)};
    };
  return CheckpointTensorImpl::make("aten::gelu_outf", rt, {self, out})[0];
}


/// ['aten::gelu', 'at::Tensor', 'gelu', '(const at::Tensor & self, c10::string_view approximate="none")']
at::Tensor checkpoint_gelu(const at::Tensor & self, c10::string_view approximate) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::gelu(vec.at(0), approximate)};
    };
  return CheckpointTensorImpl::make("aten::gelu", rt, {self})[0];
}

////////////////////////////////// auto generate part //////////////////////////////////////

/// ['aten::uniform.out', 'at::Tensor &', 'uniform_out', '(at::Tensor & out, const at::Tensor & self, double from=0, double to=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_uniform_out(at::Tensor & out, const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::uniform_out(out, vec.at(1), from, to, generator)};
    };
  return CheckpointTensorImpl::make("aten::uniform.out", rt, {out, self})[0];
}

/// ['aten::uniform.out', 'at::Tensor &', 'uniform_outf', '(const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_uniform_outf(const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::uniform_out(out, vec.at(0), from, to, generator)};
    };
  return CheckpointTensorImpl::make("aten::uniform.out", rt, {self, out})[0];
}

/// ['aten::uniform', 'at::Tensor', 'uniform', '(const at::Tensor & self, double from=0, double to=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_uniform(const at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::uniform(vec.at(0), from, to, generator)};
    };
  return CheckpointTensorImpl::make("aten::uniform", rt, {self})[0];
}

/// ['aten::copy_', 'at::Tensor &', 'copy_', '(at::Tensor & self, const at::Tensor & src, bool non_blocking)']
at::Tensor & checkpoint_copy_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  mutate_function_t mt =
    [=](const Tensors& vec) {
      Tensor self = vec.at(0);
      self.copy_(vec.at(1), non_blocking);
    };
  // Tensors inputs;
  // inputs.push_back(self);
  // inputs.push_back(src);
  CheckpointTensorImpl::mutate("copy_", mt, {self, src}, {0});
  return {self};
}

/// ['aten::copy', 'at::Tensor', 'copy', '(const at::Tensor & self, const at::Tensor & src, bool non_blocking=false)']
at::Tensor checkpoint_copy(const at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::copy(vec.at(0), vec.at(1), non_blocking)};
    };
  return CheckpointTensorImpl::make("aten::copy", rt, {self, src})[0];
}

/// ['aten::copy.out', 'at::Tensor &', 'copy_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & src, bool non_blocking=false)']
at::Tensor & checkpoint_copy_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::copy_out(out, vec.at(1), vec.at(2), non_blocking)};
    };
  return CheckpointTensorImpl::make("aten::copy.out", rt, {out, self, src})[0];
}

/// ['aten::copy.out', 'at::Tensor &', 'copy_outf', '(const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out)']
at::Tensor & checkpoint_copy_outf(const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::copy_out(out, vec.at(0), vec.at(1), non_blocking)};
    };
  return CheckpointTensorImpl::make("aten::copy.out", rt, {self, src, out})[0];
}

/// ['aten::transpose.int', 'at::Tensor', 'transpose', '(const at::Tensor & self, int64_t dim0, int64_t dim1)']
at::Tensor checkpoint_transpose(const at::Tensor & self, int64_t dim0, int64_t dim1) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::transpose(vec.at(0), dim0, dim1)};
    };
  return CheckpointTensorImpl::make("aten::transpose.int", rt, {self})[0];
}

/// ['aten::transpose.Dimname', 'at::Tensor', 'transpose', '(const at::Tensor & self, at::Dimname dim0, at::Dimname dim1)']
at::Tensor checkpoint_transpose(const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::transpose(vec.at(0), dim0, dim1)};
    };
  return CheckpointTensorImpl::make("aten::transpose.Dimname", rt, {self})[0];
}

/// ['aten::matmul', 'at::Tensor', 'matmul', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_matmul(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::matmul(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::matmul", rt, {self, other})[0];
}

/// ['aten::matmul.out', 'at::Tensor &', 'matmul_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_matmul_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::matmul_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::matmul.out", rt, {out, self, other})[0];
}

/// ['aten::matmul.out', 'at::Tensor &', 'matmul_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_matmul_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::matmul_out(out, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::matmul.out", rt, {self, other, out})[0];
}

/// ['aten::linear', 'at::Tensor', 'linear', '(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={})']
at::Tensor checkpoint_linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::linear(vec.at(0), vec.at(1), vec.at(2))};
    };
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
    const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::linear", rt, {input, weight, bias_})[0];
}

/// ['aten::linear.out', 'at::Tensor &', 'linear_out', '(at::Tensor & out, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias={})']
at::Tensor & checkpoint_linear_out(at::Tensor & out, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::linear_out(out, vec.at(1), vec.at(2), vec.at(3))};
    };
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
    const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::linear.out", rt, {out, input, weight, bias_})[0];
}

/// ['aten::linear.out', 'at::Tensor &', 'linear_outf', '(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::Tensor & out)']
at::Tensor & checkpoint_linear_outf(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(3);
      return {at::linear_out(out, vec.at(0), vec.at(1), vec.at(2))};
    };
    c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
    const Tensor& bias_ = *bias_maybe_owned;
  return CheckpointTensorImpl::make("aten::linear.out", rt, {input, weight, bias_, out})[0];
}

/// ['aten::mm', 'at::Tensor', 'mm', '(const at::Tensor & self, const at::Tensor & mat2)']
at::Tensor checkpoint_mm(const at::Tensor & self, const at::Tensor & mat2) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mm(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::mm", rt, {self, mat2})[0];
}

/// ['aten::mm.out', 'at::Tensor &', 'mm_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat2)']
at::Tensor & checkpoint_mm_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mat2) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::mm_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::mm.out", rt, {out, self, mat2})[0];
}

/// ['aten::mm.out', 'at::Tensor &', 'mm_outf', '(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out)']
at::Tensor & checkpoint_mm_outf(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::mm_out(out, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::mm.out", rt, {self, mat2, out})[0];
}

/// ['aten::normal_functional', 'at::Tensor', 'normal_functional', '(const at::Tensor & self, double mean=0, double std=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_normal_functional(const at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal_functional(vec.at(0), mean, std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal_functional", rt, {self})[0];
}

/// ['aten::normal.Tensor_float_out', 'at::Tensor &', 'normal_out', '(at::Tensor & out, const at::Tensor & mean, double std=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_out(at::Tensor & out, const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, vec.at(1), std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_float_out", rt, {out, mean})[0];
}

/// ['aten::normal.Tensor_float_out', 'at::Tensor &', 'normal_outf', '(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_outf(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::normal_out(out, vec.at(0), std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_float_out", rt, {mean, out})[0];
}

/// ['aten::normal.Tensor_float', 'at::Tensor', 'normal', '(const at::Tensor & mean, double std=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_normal(const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal(vec.at(0), std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_float", rt, {mean})[0];
}

/// ['aten::normal.float_Tensor_out', 'at::Tensor &', 'normal_out', '(at::Tensor & out, double mean, const at::Tensor & std, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_out(at::Tensor & out, double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, mean, vec.at(1), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_Tensor_out", rt, {out, std})[0];
}

/// ['aten::normal.float_Tensor_out', 'at::Tensor &', 'normal_outf', '(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_outf(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::normal_out(out, mean, vec.at(0), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_Tensor_out", rt, {std, out})[0];
}

/// ['aten::normal.float_Tensor', 'at::Tensor', 'normal', '(double mean, const at::Tensor & std, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_normal(double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal(mean, vec.at(0), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_Tensor", rt, {std})[0];
}

/// ['aten::normal.Tensor_Tensor_out', 'at::Tensor &', 'normal_out', '(at::Tensor & out, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_out(at::Tensor & out, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, vec.at(1), vec.at(2), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_Tensor_out", rt, {out, mean, std})[0];
}

/// ['aten::normal.Tensor_Tensor_out', 'at::Tensor &', 'normal_outf', '(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_outf(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::normal_out(out, vec.at(0), vec.at(1), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_Tensor_out", rt, {mean, std, out})[0];
}

/// ['aten::normal.Tensor_Tensor', 'at::Tensor', 'normal', '(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_normal(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal(vec.at(0), vec.at(1), generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.Tensor_Tensor", rt, {mean, std})[0];
}

/// ['aten::normal.float_float', 'at::Tensor', 'normal', '(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator=c10::nullopt, at::TensorOptions options={})']
at::Tensor checkpoint_normal(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal(mean, std, size_, generator, options)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float", rt, {})[0];
}

/// ['aten::normal.float_float', 'at::Tensor', 'normal', '(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_normal(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal(mean, std, size_, generator, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float", rt, {})[0];
}

/// ['aten::normal.float_float', 'at::Tensor', 'normal_symint', '(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator=c10::nullopt, at::TensorOptions options={})']
at::Tensor checkpoint_normal_symint(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal_symint(mean, std, size_, generator, options)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float", rt, {})[0];
}

/// ['aten::normal.float_float', 'at::Tensor', 'normal_symint', '(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_normal_symint(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::normal_symint(mean, std, size_, generator, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float", rt, {})[0];
}

/// ['aten::normal.float_float_out', 'at::Tensor &', 'normal_out', '(at::Tensor & out, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_out(at::Tensor & out, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, mean, std, size_, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float_out", rt, {out})[0];
}

/// ['aten::normal.float_float_out', 'at::Tensor &', 'normal_outf', '(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_outf(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, mean, std, size_, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float_out", rt, {out})[0];
}

/// ['aten::normal.float_float_out', 'at::Tensor &', 'normal_symint_out', '(at::Tensor & out, double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_symint_out(at::Tensor & out, double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_symint_out(out, mean, std, size_, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float_out", rt, {out})[0];
}

/// ['aten::normal.float_float_out', 'at::Tensor &', 'normal_symint_outf', '(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_symint_outf(double mean, double std, c10::SymIntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_symint_out(out, mean, std, size_, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.float_float_out", rt, {out})[0];
}

/// ['aten::normal.out', 'at::Tensor &', 'normal_out', '(at::Tensor & out, const at::Tensor & self, double mean=0, double std=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_normal_out(at::Tensor & out, const at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::normal_out(out, vec.at(1), mean, std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.out", rt, {out, self})[0];
}

/// ['aten::normal.out', 'at::Tensor &', 'normal_outf', '(const at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_normal_outf(const at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::normal_out(out, vec.at(0), mean, std, generator)};
    };
  return CheckpointTensorImpl::make("aten::normal.out", rt, {self, out})[0];
}

/// ['aten::index.Tensor', 'at::Tensor', 'index', '(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices)']
/*
bool tensorlist_has_dispatch(const c10::List<c10::optional<at::Tensor>>& li) {
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    if (t && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  return false;
}

auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  at::cuda::index_out(
      result, *std::get<1>(mask_self_expanded),
      c10::List<c10::optional<at::Tensor>>({*std::move(std::get<0>(mask_self_expanded))}));
*/
at::Tensor checkpoint_index(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto self = vec.at(0);
      c10::List<c10::optional<at::Tensor>> inds;
      for(int i=1; i<vec.size(); i++){
        c10::optional<at::Tensor> t_(vec.at(i));
        inds.push_back(t_);
      }
      return {at::index(vec.at(0), inds)};
    };
  std::vector<Tensor> inputs = {self};
  for (auto i : c10::irange(indices.size())) {
    auto t = indices.get(i);
    c10::MaybeOwned<Tensor> t_maybe_owned = at::borrow_from_optional_tensor(t);
    const Tensor& t_ = *t_maybe_owned;
    inputs.push_back(t_);
  }
  return CheckpointTensorImpl::make("aten::index.Tensor", rt, inputs)[0];
}

/// ['aten::index.Tensor_out', 'at::Tensor &', 'index_out', '(at::Tensor & out, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices)']
// at::Tensor & checkpoint_index_out(at::Tensor & out, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor out = vec.at(0);
//       return {at::index_out(out, vec.at(1), vec.at(2))};
//     };
//     c10::MaybeOwned<Tensor> indices_maybe_owned = at::borrow_from_optional_tensor(indices);
//     const Tensor& indices_ = *indices_maybe_owned;
//   return CheckpointTensorImpl::make("aten::index.Tensor_out", rt, {out, self, indices_})[0];
// }

/// ['aten::index.Tensor_out', 'at::Tensor &', 'index_outf', '(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, at::Tensor & out)']
// at::Tensor & checkpoint_index_outf(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, at::Tensor & out) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor out = vec.at(2);
//       return {at::index_outf(vec.at(0), vec.at(1), out)};
//     };
//     c10::MaybeOwned<Tensor> indices_maybe_owned = at::borrow_from_optional_tensor(indices);
//     const Tensor& indices_ = *indices_maybe_owned;
//   return CheckpointTensorImpl::make("aten::index.Tensor_out", rt, {self, indices_, out})[0];
// }

/// ['aten::_to_copy', 'at::Tensor', '_to_copy', '(const at::Tensor & self, at::TensorOptions options={}, bool non_blocking=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
// at::Tensor checkpoint__to_copy(const at::Tensor & self, at::TensorOptions options, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::_to_copy(vec.at(0), options, non_blocking, memory_format)};
//     };
//   return CheckpointTensorImpl::make("aten::_to_copy", rt, {self})[0];
// }

/// ['aten::_to_copy', 'at::Tensor', '_to_copy', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format)']
// at::Tensor checkpoint__to_copy(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::_to_copy(vec.at(0), dtype, layout, device, pin_memory, non_blocking, memory_format)};
//     };
//   return CheckpointTensorImpl::make("aten::_to_copy", rt, {self})[0];
// }

/// ['aten::_to_copy.out', 'at::Tensor &', '_to_copy_out', '(at::Tensor & out, const at::Tensor & self, bool non_blocking=false, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint__to_copy_out(at::Tensor & out, const at::Tensor & self, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::_to_copy_out(out, vec.at(1), non_blocking, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::_to_copy.out", rt, {out, self})[0];
}

/// ['aten::_to_copy.out', 'at::Tensor &', '_to_copy_outf', '(const at::Tensor & self, bool non_blocking, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint__to_copy_outf(const at::Tensor & self, bool non_blocking, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::_to_copy_out(out, vec.at(0), non_blocking, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::_to_copy.out", rt, {self, out})[0];
}

/// ['aten::slice.Tensor', 'at::Tensor', 'slice', '(const at::Tensor & self, int64_t dim=0, c10::optional<int64_t> start=c10::nullopt, c10::optional<int64_t> end=c10::nullopt, int64_t step=1)']
at::Tensor checkpoint_slice(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice(vec.at(0), dim, start, end, step)};
    };
  return CheckpointTensorImpl::make("aten::slice.Tensor", rt, {self})[0];
}

/// ['aten::slice.Tensor', 'at::Tensor', 'slice_symint', '(const at::Tensor & self, int64_t dim=0, c10::optional<c10::SymInt> start=c10::nullopt, c10::optional<c10::SymInt> end=c10::nullopt, c10::SymInt step=1)']
at::Tensor checkpoint_slice_symint(const at::Tensor & self, int64_t dim, c10::optional<c10::SymInt> start, c10::optional<c10::SymInt> end, c10::SymInt step) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::slice_symint(vec.at(0), dim, start, end, step)};
    };
  return CheckpointTensorImpl::make("aten::slice.Tensor", rt, {self})[0];
}

/// ['aten::reshape', 'at::Tensor', 'reshape', '(const at::Tensor & self, at::IntArrayRef shape)']
at::Tensor checkpoint_reshape(const at::Tensor & self, at::IntArrayRef shape) {
  auto shape_ = shape.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::reshape(vec.at(0), shape_)};
    };
  return CheckpointTensorImpl::make("aten::reshape", rt, {self})[0];
}

/// ['aten::reshape', 'at::Tensor', 'reshape_symint', '(const at::Tensor & self, c10::SymIntArrayRef shape)']
at::Tensor checkpoint_reshape_symint(const at::Tensor & self, c10::SymIntArrayRef shape) {
  auto shape_ = shape.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::reshape_symint(vec.at(0), shape_)};
    };
  return CheckpointTensorImpl::make("aten::reshape", rt, {self})[0];
}

/// ['aten::mul.Tensor', 'at::Tensor', 'mul', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_mul(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::mul.Tensor", rt, {self, other})[0];
}

/// ['aten::mul.out', 'at::Tensor &', 'mul_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_mul_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::mul_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::mul.out", rt, {out, self, other})[0];
}

/// ['aten::mul.out', 'at::Tensor &', 'mul_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_mul_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::mul_out(out, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::mul.out", rt, {self, other, out})[0];
}

/// ['aten::mul.Scalar', 'at::Tensor', 'mul', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_mul(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mul(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::mul.Scalar", rt, {self})[0];
}

/// ['aten::mul.Scalar_out', 'at::Tensor &', 'mul_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_mul_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::mul_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::mul.Scalar_out", rt, {out, self})[0];
}

/// ['aten::mul.Scalar_out', 'at::Tensor &', 'mul_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_mul_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::mul_out(out,vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::mul.Scalar_out", rt, {self, out})[0];
}

/// ['aten::t', 'at::Tensor', 't', '(const at::Tensor & self)']
at::Tensor checkpoint_t(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::t(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::t", rt, {self})[0];
}

/// ['aten::_flash_attention_forward', '::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>', '_flash_attention_forward', '(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, bool return_debug_mask, c10::optional<double> scale=c10::nullopt)']
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> checkpoint__flash_attention_forward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, bool return_debug_mask, c10::optional<double> scale) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::_flash_attention_forward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), max_q, max_k, dropout_p, is_causal, return_debug_mask, scale);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret), std::get<3>(ret), std::get<4>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_flash_attention_forward", rt, {query, key, value, cum_seq_q, cum_seq_k});
  return {ret[0], ret[1], ret[2], ret[3], ret[4]};
}

/// ['aten::_flash_attention_backward', '::std::tuple<at::Tensor,at::Tensor,at::Tensor>', '_flash_attention_backward', '(const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, const at::Tensor & philox_seed, const at::Tensor & philox_offset, c10::optional<double> scale=c10::nullopt)']
::std::tuple<at::Tensor,at::Tensor,at::Tensor> checkpoint__flash_attention_backward(const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, const at::Tensor & philox_seed, const at::Tensor & philox_offset, c10::optional<double> scale) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::_flash_attention_backward(vec.at(0), vec.at(1), vec.at(2), vec.at(3), vec.at(4), vec.at(5), vec.at(6), vec.at(7), max_q, max_k, dropout_p, is_causal, vec.at(8), vec.at(9), scale);
      return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::_flash_attention_backward", rt, {grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, philox_seed, philox_offset});
  return {ret[0], ret[1], ret[2]};
}

/// ['aten::pow.Tensor_Tensor_out', 'at::Tensor &', 'pow_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & exponent)']
at::Tensor & checkpoint_pow_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::pow_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Tensor_out", rt, {out, self, exponent})[0];
}

/// ['aten::pow.Tensor_Tensor_out', 'at::Tensor &', 'pow_outf', '(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out)']
at::Tensor & checkpoint_pow_outf(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::pow_out(out, vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Tensor_out", rt, {self, exponent, out})[0];
}

/// ['aten::pow.Tensor_Tensor', 'at::Tensor', 'pow', '(const at::Tensor & self, const at::Tensor & exponent)']
at::Tensor checkpoint_pow(const at::Tensor & self, const at::Tensor & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::pow(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Tensor", rt, {self, exponent})[0];
}

/// ['aten::pow.Scalar_out', 'at::Tensor &', 'pow_out', '(at::Tensor & out, const at::Scalar & self, const at::Tensor & exponent)']
at::Tensor & checkpoint_pow_out(at::Tensor & out, const at::Scalar & self, const at::Tensor & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::pow_out(out, self, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::pow.Scalar_out", rt, {out, exponent})[0];
}

/// ['aten::pow.Scalar_out', 'at::Tensor &', 'pow_outf', '(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out)']
at::Tensor & checkpoint_pow_outf(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::pow_out(out, self, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::pow.Scalar_out", rt, {exponent, out})[0];
}

/// ['aten::pow.Scalar', 'at::Tensor', 'pow', '(const at::Scalar & self, const at::Tensor & exponent)']
at::Tensor checkpoint_pow(const at::Scalar & self, const at::Tensor & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::pow(self, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::pow.Scalar", rt, {exponent})[0];
}

/// ['aten::pow.Tensor_Scalar_out', 'at::Tensor &', 'pow_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & exponent)']
at::Tensor & checkpoint_pow_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::pow_out(out, vec.at(1), exponent)};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Scalar_out", rt, {out, self})[0];
}

/// ['aten::pow.Tensor_Scalar_out', 'at::Tensor &', 'pow_outf', '(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out)']
at::Tensor & checkpoint_pow_outf(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::pow_out(out, vec.at(0), exponent)};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Scalar_out", rt, {self, out})[0];
}

/// ['aten::pow.Tensor_Scalar', 'at::Tensor', 'pow', '(const at::Tensor & self, const at::Scalar & exponent)']
at::Tensor checkpoint_pow(const at::Tensor & self, const at::Scalar & exponent) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::pow(vec.at(0), exponent)};
    };
  return CheckpointTensorImpl::make("aten::pow.Tensor_Scalar", rt, {self})[0];
}

/// ['aten::add.Tensor', 'at::Tensor', 'add', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [alpha](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), vec.at(1), alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.Tensor", rt, {self, other})[0];
}

/// ['aten::add.out', 'at::Tensor &', 'add_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::add_out(out, vec.at(1), vec.at(2), alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.out", rt, {out, self, other})[0];
}

/// ['aten::add.out', 'at::Tensor &', 'add_outf', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_add_outf(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::add_out(out, vec.at(0), vec.at(1), alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.out", rt, {self, other, out})[0];
}

/// ['aten::add.Scalar', 'at::Tensor', 'add', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_add(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [other, alpha](const Tensors& vec) -> Tensors {
      return {at::add(vec.at(0), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.Scalar", rt, {self})[0];
}

/// ['aten::add.Scalar_out', 'at::Tensor &', 'add_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_add_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::add_out(out, vec.at(1), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.Scalar_out", rt, {out, self})[0];
}

/// ['aten::add.Scalar_out', 'at::Tensor &', 'add_outf', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_add_outf(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::add_out(out, vec.at(0), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::add.Scalar_out", rt, {self, out})[0];
}

/// ['aten::squeeze', 'at::Tensor', 'squeeze', '(const at::Tensor & self)']
at::Tensor checkpoint_squeeze(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::squeeze(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::squeeze", rt, {self})[0];
}

/// ['aten::squeeze.dim', 'at::Tensor', 'squeeze', '(const at::Tensor & self, int64_t dim)']
at::Tensor checkpoint_squeeze(const at::Tensor & self, int64_t dim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::squeeze(vec.at(0), dim)};
    };
  return CheckpointTensorImpl::make("aten::squeeze.dim", rt, {self})[0];
}

/// ['aten::squeeze.dimname', 'at::Tensor', 'squeeze', '(const at::Tensor & self, at::Dimname dim)']
at::Tensor checkpoint_squeeze(const at::Tensor & self, at::Dimname dim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::squeeze(vec.at(0), dim)};
    };
  return CheckpointTensorImpl::make("aten::squeeze.dimname", rt, {self})[0];
}

/// ['aten::squeeze.dims', 'at::Tensor', 'squeeze', '(const at::Tensor & self, at::IntArrayRef dim)']
at::Tensor checkpoint_squeeze(const at::Tensor & self, at::IntArrayRef dim) {
  auto dim_ = dim.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::squeeze(vec.at(0), dim_)};
    };
  return CheckpointTensorImpl::make("aten::squeeze.dims", rt, {self})[0];
}

/// ['aten::cat', 'at::Tensor', 'cat', '(const at::ITensorListRef & tensors, int64_t dim=0)']
at::Tensor checkpoint_cat(const at::ITensorListRef & tensors, int64_t dim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cat(vec, dim)};
    };
  std::vector<Tensor> inputs;
  for (const auto& t : tensors) {
    inputs.push_back(t);
  }
  return CheckpointTensorImpl::make("aten::cat", rt, inputs)[0];
}

/// ['aten::cat.out', 'at::Tensor &', 'cat_out', '(at::Tensor & out, const at::ITensorListRef & tensors, int64_t dim=0)']
// at::Tensor & checkpoint_cat_out(at::Tensor & out, const at::ITensorListRef & tensors, int64_t dim) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor out = vec.at(0);
//       return {at::cat_out(out, vec.at(1), dim)};
//     };
//   return CheckpointTensorImpl::make("aten::cat.out", rt, {out, tensors})[0];
// }

/// ['aten::cat.out', 'at::Tensor &', 'cat_outf', '(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out)']
at::Tensor & checkpoint_cat_outf(const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::cat_out(out, std::vector<Tensor>(vec.begin() + 1, vec.end()), dim)};
    };
  std::vector<Tensor> inputs;
  inputs.push_back(out);
  for (const auto& t : tensors) {
    inputs.push_back(t);
  }
  return CheckpointTensorImpl::make("aten::cat.out", rt, inputs)[0];
}

/// ['aten::cat.names', 'at::Tensor', 'cat', '(at::TensorList tensors, at::Dimname dim)']
// at::Tensor checkpoint_cat(at::TensorList tensors, at::Dimname dim) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor tensors = vec.at(0);
//       return {at::cat(tensors, dim)};
//     };
//   return CheckpointTensorImpl::make("aten::cat.names", rt, {tensors})[0];
// }

/// ['aten::cat.names_out', 'at::Tensor &', 'cat_out', '(at::Tensor & out, at::TensorList tensors, at::Dimname dim)']
// at::Tensor & checkpoint_cat_out(at::Tensor & out, at::TensorList tensors, at::Dimname dim) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor out = vec.at(0);
//       Tensor tensors = vec.at(1);
//       return {at::cat_out(out, tensors, dim)};
//     };
//   return CheckpointTensorImpl::make("aten::cat.names_out", rt, {out, tensors})[0];
// }

/// ['aten::cat.names_out', 'at::Tensor &', 'cat_outf', '(at::TensorList tensors, at::Dimname dim, at::Tensor & out)']
// at::Tensor & checkpoint_cat_outf(at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor tensors = vec.at(0);
//       Tensor out = vec.at(1);
//       return {at::cat_outf(tensors, dim, out)};
//     };
//   return CheckpointTensorImpl::make("aten::cat.names_out", rt, {tensors, out})[0];
// }

/// ['aten::empty_strided', 'at::Tensor', 'empty_strided', '(at::IntArrayRef size, at::IntArrayRef stride, at::TensorOptions options={})']
at::Tensor checkpoint_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, at::TensorOptions options) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_strided(size_, stride_, options)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided", rt, {})[0];
}

/// ['aten::empty_strided', 'at::Tensor', 'empty_strided', '(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_strided(size_, stride_, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided", rt, {})[0];
}

/// ['aten::empty_strided', 'at::Tensor', 'empty_strided_symint', '(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options={})']
at::Tensor checkpoint_empty_strided_symint(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::TensorOptions options) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_strided_symint(size_, stride_, options)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided", rt, {})[0];
}

/// ['aten::empty_strided', 'at::Tensor', 'empty_strided_symint', '(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_empty_strided_symint(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_strided_symint(size_, stride_, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided", rt, {})[0];
}

/// ['aten::empty_strided.out', 'at::Tensor &', 'empty_strided_out', '(at::Tensor & out, at::IntArrayRef size, at::IntArrayRef stride)']
at::Tensor & checkpoint_empty_strided_out(at::Tensor & out, at::IntArrayRef size, at::IntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_strided_out(out, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided.out", rt, {out})[0];
}

/// ['aten::empty_strided.out', 'at::Tensor &', 'empty_strided_outf', '(at::IntArrayRef size, at::IntArrayRef stride, at::Tensor & out)']
at::Tensor & checkpoint_empty_strided_outf(at::IntArrayRef size, at::IntArrayRef stride, at::Tensor & out) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_strided_out(out, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided.out", rt, {out})[0];
}

/// ['aten::empty_strided.out', 'at::Tensor &', 'empty_strided_symint_out', '(at::Tensor & out, c10::SymIntArrayRef size, c10::SymIntArrayRef stride)']
at::Tensor & checkpoint_empty_strided_symint_out(at::Tensor & out, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_strided_symint_out(out, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided.out", rt, {out})[0];
}

/// ['aten::empty_strided.out', 'at::Tensor &', 'empty_strided_symint_outf', '(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out)']
at::Tensor & checkpoint_empty_strided_symint_outf(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_strided_symint_out(out, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::empty_strided.out", rt, {out})[0];
}

/// ['aten::unsqueeze', 'at::Tensor', 'unsqueeze', '(const at::Tensor & self, int64_t dim)']
at::Tensor checkpoint_unsqueeze(const at::Tensor & self, int64_t dim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::unsqueeze(vec.at(0), dim)};
    };
  return CheckpointTensorImpl::make("aten::unsqueeze", rt, {self})[0];
}

/// ['aten::mean', 'at::Tensor', 'mean', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_mean(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mean(vec.at(0), dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean", rt, {self})[0];
}

/// ['aten::mean.dim', 'at::Tensor', 'mean', '(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_mean(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      if(dim.has_value())
        return {at::mean(vec.at(0), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype)};
      else
        return {at::mean(vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.dim", rt, {self})[0];
}

/// ['aten::mean.out', 'at::Tensor &', 'mean_out', '(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_mean_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      if(dim.has_value())
        return {at::mean_out(out, vec.at(1), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype)};
      else
        return {at::mean_out(out, vec.at(1), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.out", rt, {out, self})[0];
}

/// ['aten::mean.out', 'at::Tensor &', 'mean_outf', '(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_mean_outf(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      if(dim.has_value())
        return {at::mean_out(out, vec.at(0), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype)};
      else
        return {at::mean_out(out, vec.at(0), dim_, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.out", rt, {self, out})[0];
}

/// ['aten::mean.names_dim', 'at::Tensor', 'mean', '(const at::Tensor & self, at::DimnameList dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_mean(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::mean(vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.names_dim", rt, {self})[0];
}

/// ['aten::mean.names_out', 'at::Tensor &', 'mean_out', '(at::Tensor & out, const at::Tensor & self, at::DimnameList dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_mean_out(at::Tensor & out, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::mean_out(out, vec.at(1), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.names_out", rt, {out, self})[0];
}

/// ['aten::mean.names_out', 'at::Tensor &', 'mean_outf', '(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_mean_outf(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::mean_out(out, vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::mean.names_out", rt, {self, out})[0];
}

/// ['aten::neg', 'at::Tensor', 'neg', '(const at::Tensor & self)']
at::Tensor checkpoint_neg(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::neg(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::neg", rt, {self})[0];
}

/// ['aten::neg_', 'at::Tensor &', 'neg_', '(at::Tensor & self)']
at::Tensor & checkpoint_neg_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::neg_(self)};
    };
  return CheckpointTensorImpl::make("aten::neg_", rt, {self})[0];
}

/// ['aten::neg.out', 'at::Tensor &', 'neg_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_neg_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::neg_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::neg.out", rt, {out, self})[0];
}

/// ['aten::neg.out', 'at::Tensor &', 'neg_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_neg_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::neg_out(out, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::neg.out", rt, {self, out})[0];
}

/// ['aten::rsqrt', 'at::Tensor', 'rsqrt', '(const at::Tensor & self)']
// at::Tensor checkpoint_rsqrt(const at::Tensor & self) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::rsqrt(vec.at(0))};
//     };
//   return CheckpointTensorImpl::make("aten::rsqrt", rt, {self})[0];
// }

/// ['aten::rsqrt_', 'at::Tensor &', 'rsqrt_', '(at::Tensor & self)']
at::Tensor & checkpoint_rsqrt_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::rsqrt_(self)};
    };
  return CheckpointTensorImpl::make("aten::rsqrt_", rt, {self})[0];
}

/// ['aten::rsqrt.out', 'at::Tensor &', 'rsqrt_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_rsqrt_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::rsqrt_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::rsqrt.out", rt, {out, self})[0];
}

/// ['aten::rsqrt.out', 'at::Tensor &', 'rsqrt_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_rsqrt_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::rsqrt_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::rsqrt.out", rt, {self, out})[0];
}

/// ['aten::as_strided', 'at::Tensor', 'as_strided', '(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset=c10::nullopt)']
at::Tensor checkpoint_as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided(vec.at(0), size_, stride_, storage_offset)};
    };
  return CheckpointTensorImpl::make("aten::as_strided", rt, {self})[0];
}

/// ['aten::as_strided', 'at::Tensor', 'as_strided_symint', '(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset=c10::nullopt)']
at::Tensor checkpoint_as_strided_symint(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::as_strided_symint(vec.at(0), size_, stride_, storage_offset)};
    };
  return CheckpointTensorImpl::make("aten::as_strided", rt, {self})[0];
}

/// ['aten::empty_like', 'at::Tensor', 'empty_like', '(const at::Tensor & self, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_empty_like(const at::Tensor & self, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_like(vec.at(0), options, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty_like", rt, {self})[0];
}

/// ['aten::empty_like', 'at::Tensor', 'empty_like', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)']
at::Tensor checkpoint_empty_like(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_like(vec.at(0), dtype, layout, device, pin_memory, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty_like", rt, {self})[0];
}

/// ['aten::empty_like.out', 'at::Tensor &', 'empty_like_out', '(at::Tensor & out, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint_empty_like_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_like_out(out, vec.at(1), memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty_like.out", rt, {out, self})[0];
}

/// ['aten::empty_like.out', 'at::Tensor &', 'empty_like_outf', '(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint_empty_like_outf(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::empty_like_out(out, vec.at(0), memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty_like.out", rt, {self, out})[0];
}

/// ['aten::empty.names', 'at::Tensor', 'empty', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_empty(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty(size_, names, options, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.names", rt, {})[0];
}

/// ['aten::empty.names', 'at::Tensor', 'empty', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)']
at::Tensor checkpoint_empty(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty(size_, names, dtype, layout, device, pin_memory, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.names", rt, {})[0];
}

/// ['aten::empty.memory_format', 'at::Tensor', 'empty', '(at::IntArrayRef size, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_empty(at::IntArrayRef size, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty(size_, options, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.memory_format", rt, {})[0];
}

/// ['aten::empty.memory_format', 'at::Tensor', 'empty', '(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)']
at::Tensor checkpoint_empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty(size_, dtype, layout, device, pin_memory, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.memory_format", rt, {})[0];
}

/// ['aten::empty.memory_format', 'at::Tensor', 'empty_symint', '(c10::SymIntArrayRef size, at::TensorOptions options={}, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_empty_symint(c10::SymIntArrayRef size, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_symint(size_, options, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.memory_format", rt, {})[0];
}

/// ['aten::empty.memory_format', 'at::Tensor', 'empty_symint', '(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)']
at::Tensor checkpoint_empty_symint(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::empty_symint(size_, dtype, layout, device, pin_memory, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.memory_format", rt, {})[0];
}

/// ['aten::empty.out', 'at::Tensor &', 'empty_out', '(at::Tensor & out, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint_empty_out(at::Tensor & out, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_out(out, size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.out", rt, {out})[0];
}

/// ['aten::empty.out', 'at::Tensor &', 'empty_outf', '(at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint_empty_outf(at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_out(out, size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.out", rt, {out})[0];
}

/// ['aten::empty.out', 'at::Tensor &', 'empty_symint_out', '(at::Tensor & out, c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint_empty_symint_out(at::Tensor & out, c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_symint_out(out, size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.out", rt, {out})[0];
}

/// ['aten::empty.out', 'at::Tensor &', 'empty_symint_outf', '(c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint_empty_symint_outf(c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_symint_out(out, size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.out", rt, {out})[0];
}

/// ['aten::empty.names_out', 'at::Tensor &', 'empty_out', '(at::Tensor & out, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint_empty_out(at::Tensor & out, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_out(out, size_, names, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.names_out", rt, {out})[0];
}

/// ['aten::empty.names_out', 'at::Tensor &', 'empty_outf', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint_empty_outf(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::empty_out(out, size_, names, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::empty.names_out", rt, {out})[0];
}

/// ['aten::silu', 'at::Tensor', 'silu', '(const at::Tensor & self)']
at::Tensor checkpoint_silu(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::silu(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::silu", rt, {self})[0];
}

/// ['aten::silu_', 'at::Tensor &', 'silu_', '(at::Tensor & self)']
at::Tensor & checkpoint_silu_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::silu_(self)};
    };
  return CheckpointTensorImpl::make("aten::silu_", rt, {self})[0];
}

/// ['aten::silu.out', 'at::Tensor &', 'silu_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_silu_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::silu_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::silu.out", rt, {out, self})[0];
}

/// ['aten::silu.out', 'at::Tensor &', 'silu_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_silu_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::silu_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::silu.out", rt, {self, out})[0];
}

/// ['aten::select.Dimname', 'at::Tensor', 'select', '(const at::Tensor & self, at::Dimname dim, int64_t index)']
at::Tensor checkpoint_select(const at::Tensor & self, at::Dimname dim, int64_t index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select(vec.at(0), dim, index)};
    };
  return CheckpointTensorImpl::make("aten::select.Dimname", rt, {self})[0];
}

/// ['aten::select.int', 'at::Tensor', 'select', '(const at::Tensor & self, int64_t dim, int64_t index)']
at::Tensor checkpoint_select(const at::Tensor & self, int64_t dim, int64_t index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select(vec.at(0), dim, index)};
    };
  return CheckpointTensorImpl::make("aten::select.int", rt, {self})[0];
}

/// ['aten::select.int', 'at::Tensor', 'select_symint', '(const at::Tensor & self, int64_t dim, c10::SymInt index)']
at::Tensor checkpoint_select_symint(const at::Tensor & self, int64_t dim, c10::SymInt index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::select_symint(vec.at(0), dim, index)};
    };
  return CheckpointTensorImpl::make("aten::select.int", rt, {self})[0];
}

/// ['aten::_unsafe_view', 'at::Tensor', '_unsafe_view', '(const at::Tensor & self, at::IntArrayRef size)']
at::Tensor checkpoint__unsafe_view(const at::Tensor & self, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_unsafe_view(vec.at(0), size_)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view", rt, {self})[0];
}

/// ['aten::_unsafe_view', 'at::Tensor', '_unsafe_view_symint', '(const at::Tensor & self, c10::SymIntArrayRef size)']
at::Tensor checkpoint__unsafe_view_symint(const at::Tensor & self, c10::SymIntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_unsafe_view_symint(vec.at(0), size_)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view", rt, {self})[0];
}

/// ['aten::_unsafe_view.out', 'at::Tensor &', '_unsafe_view_out', '(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size)']
at::Tensor & checkpoint__unsafe_view_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::_unsafe_view_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view.out", rt, {out, self})[0];
}

/// ['aten::_unsafe_view.out', 'at::Tensor &', '_unsafe_view_outf', '(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint__unsafe_view_outf(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::_unsafe_view_outf(vec.at(0), size_, out)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view.out", rt, {self, out})[0];
}

/// ['aten::_unsafe_view.out', 'at::Tensor &', '_unsafe_view_symint_out', '(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size)']
at::Tensor & checkpoint__unsafe_view_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::_unsafe_view_symint_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view.out", rt, {out, self})[0];
}

/// ['aten::_unsafe_view.out', 'at::Tensor &', '_unsafe_view_symint_outf', '(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint__unsafe_view_symint_outf(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::_unsafe_view_symint_outf(vec.at(0), size_, out)};
    };
  return CheckpointTensorImpl::make("aten::_unsafe_view.out", rt, {self, out})[0];
}

/// ['aten::detach', 'at::Tensor', 'detach', '(const at::Tensor & self)']
at::Tensor checkpoint_detach(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {vec.at(0).detach()};
    };
  return CheckpointTensorImpl::make("aten::detach", rt, {self})[0];
}

/// ['aten::detach_', 'at::Tensor &', 'detach_', '(at::Tensor & self)']
at::Tensor & checkpoint_detach_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::detach_(self)};
    };
  return CheckpointTensorImpl::make("aten::detach_", rt, {self})[0];
}

/// ['aten::multinomial.out', 'at::Tensor &', 'multinomial_out', '(at::Tensor & out, const at::Tensor & self, int64_t num_samples, bool replacement=false, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_multinomial_out(at::Tensor & out, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::multinomial_out(out, vec.at(1), num_samples, replacement, generator)};
    };
  return CheckpointTensorImpl::make("aten::multinomial.out", rt, {out, self})[0];
}

/// ['aten::multinomial.out', 'at::Tensor &', 'multinomial_outf', '(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_multinomial_outf(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::multinomial_out(out, vec.at(0), num_samples, replacement, generator)};
    };
  return CheckpointTensorImpl::make("aten::multinomial.out", rt, {self, out})[0];
}

/// ['aten::multinomial', 'at::Tensor', 'multinomial', '(const at::Tensor & self, int64_t num_samples, bool replacement=false, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_multinomial(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::multinomial(vec.at(0), num_samples, replacement, generator)};
    };
  return CheckpointTensorImpl::make("aten::multinomial", rt, {self})[0];
}

/// ['aten::cos', 'at::Tensor', 'cos', '(const at::Tensor & self)']
at::Tensor checkpoint_cos(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cos(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::cos", rt, {self})[0];
}

/// ['aten::cos_', 'at::Tensor &', 'cos_', '(at::Tensor & self)']
at::Tensor & checkpoint_cos_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::cos_(self)};
    };
  return CheckpointTensorImpl::make("aten::cos_", rt, {self})[0];
}

/// ['aten::cos.out', 'at::Tensor &', 'cos_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_cos_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::cos_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::cos.out", rt, {out, self})[0];
}

/// ['aten::cos.out', 'at::Tensor &', 'cos_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_cos_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::cos_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::cos.out", rt, {self, out})[0];
}

/// ['aten::sin', 'at::Tensor', 'sin', '(const at::Tensor & self)']
at::Tensor checkpoint_sin(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sin(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::sin", rt, {self})[0];
}

/// ['aten::sin_', 'at::Tensor &', 'sin_', '(at::Tensor & self)']
at::Tensor & checkpoint_sin_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::sin_(self)};
    };
  return CheckpointTensorImpl::make("aten::sin_", rt, {self})[0];
}

/// ['aten::sin.out', 'at::Tensor &', 'sin_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_sin_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::sin_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::sin.out", rt, {out, self})[0];
}

/// ['aten::sin.out', 'at::Tensor &', 'sin_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_sin_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::sin_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::sin.out", rt, {self, out})[0];
}

/// ['aten::sort.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim=-1, bool descending=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::sort_out(values, indices, vec.at(2), dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.values", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_outf', '(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_outf(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::sort_outf(vec.at(0), dim, descending, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.values", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::sort.values_stable', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::optional<bool> stable, int64_t dim=-1, bool descending=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::sort_out(values, indices, vec.at(2), stable, dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.values_stable", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.values_stable', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_outf', '(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_outf(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::sort_outf(vec.at(0), stable, dim, descending, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.values_stable", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::sort', '::std::tuple<at::Tensor,at::Tensor>', 'sort', '(const at::Tensor & self, int64_t dim=-1, bool descending=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_sort(const at::Tensor & self, int64_t dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::sort(vec.at(0), dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.stable', '::std::tuple<at::Tensor,at::Tensor>', 'sort', '(const at::Tensor & self, c10::optional<bool> stable, int64_t dim=-1, bool descending=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_sort(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::sort(vec.at(0), stable, dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.stable", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname_values', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool descending=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, at::Dimname dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::sort_out(values, indices, vec.at(2), dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname_values", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname_values', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_outf', '(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_outf(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::sort_outf(vec.at(0), dim, descending, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname_values", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname_values_stable', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::sort_out(values, indices, vec.at(2), stable, dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname_values_stable", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname_values_stable', '::std::tuple<at::Tensor &,at::Tensor &>', 'sort_outf', '(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_sort_outf(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::sort_outf(vec.at(0), stable, dim, descending, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname_values_stable", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname', '::std::tuple<at::Tensor,at::Tensor>', 'sort', '(const at::Tensor & self, at::Dimname dim, bool descending=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_sort(const at::Tensor & self, at::Dimname dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::sort(vec.at(0), dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::sort.dimname_stable', '::std::tuple<at::Tensor,at::Tensor>', 'sort', '(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_sort(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::sort(vec.at(0), stable, dim, descending);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::sort.dimname_stable", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::repeat.out', 'at::Tensor &', 'repeat_out', '(at::Tensor & out, const at::Tensor & self, at::IntArrayRef repeats)']
at::Tensor & checkpoint_repeat_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef repeats) {
  auto repeats_ = repeats.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::repeat_out(out, vec.at(1), repeats_)};
    };
  return CheckpointTensorImpl::make("aten::repeat.out", rt, {out, self})[0];
}

/// ['aten::repeat.out', 'at::Tensor &', 'repeat_outf', '(const at::Tensor & self, at::IntArrayRef repeats, at::Tensor & out)']
at::Tensor & checkpoint_repeat_outf(const at::Tensor & self, at::IntArrayRef repeats, at::Tensor & out) {
  auto repeats_ = repeats.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::repeat_outf(vec.at(0), repeats_, out)};
    };
  return CheckpointTensorImpl::make("aten::repeat.out", rt, {self, out})[0];
}

/// ['aten::repeat.out', 'at::Tensor &', 'repeat_symint_out', '(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef repeats)']
at::Tensor & checkpoint_repeat_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef repeats) {
  auto repeats_ = repeats.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::repeat_symint_out(out, vec.at(1), repeats_)};
    };
  return CheckpointTensorImpl::make("aten::repeat.out", rt, {out, self})[0];
}

/// ['aten::repeat.out', 'at::Tensor &', 'repeat_symint_outf', '(const at::Tensor & self, c10::SymIntArrayRef repeats, at::Tensor & out)']
at::Tensor & checkpoint_repeat_symint_outf(const at::Tensor & self, c10::SymIntArrayRef repeats, at::Tensor & out) {
  auto repeats_ = repeats.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::repeat_symint_outf(vec.at(0), repeats_, out)};
    };
  return CheckpointTensorImpl::make("aten::repeat.out", rt, {self, out})[0];
}

/// ['aten::set.source_Storage_out', 'at::Tensor &', 'set_out', '(at::Tensor & out, const at::Tensor & self, at::Storage source)']
at::Tensor & checkpoint_set_out(at::Tensor & out, const at::Tensor & self, at::Storage source) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::set_out(out, vec.at(1), source)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_out", rt, {out, self})[0];
}

/// ['aten::set.source_Storage_out', 'at::Tensor &', 'set_outf', '(const at::Tensor & self, at::Storage source, at::Tensor & out)']
at::Tensor & checkpoint_set_outf(const at::Tensor & self, at::Storage source, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::set_outf(vec.at(0), source, out)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_out", rt, {self, out})[0];
}

/// ['aten::set.source_Storage', 'at::Tensor', 'set', '(const at::Tensor & self, at::Storage source)']
at::Tensor checkpoint_set(const at::Tensor & self, at::Storage source) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::set(vec.at(0), source)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage", rt, {self})[0];
}

/// ['aten::set.source_Storage_storage_offset_out', 'at::Tensor &', 'set_out', '(at::Tensor & out, const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride={})']
at::Tensor & checkpoint_set_out(at::Tensor & out, const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::set_out(out, vec.at(1), source, storage_offset, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset_out", rt, {out, self})[0];
}

/// ['aten::set.source_Storage_storage_offset_out', 'at::Tensor &', 'set_outf', '(const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride, at::Tensor & out)']
at::Tensor & checkpoint_set_outf(const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride, at::Tensor & out) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::set_outf(vec.at(0), source, storage_offset, size_, stride_, out)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset_out", rt, {self, out})[0];
}

/// ['aten::set.source_Storage_storage_offset_out', 'at::Tensor &', 'set_symint_out', '(at::Tensor & out, const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride={})']
at::Tensor & checkpoint_set_symint_out(at::Tensor & out, const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::set_symint_out(out, vec.at(1), source, storage_offset, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset_out", rt, {out, self})[0];
}

/// ['aten::set.source_Storage_storage_offset_out', 'at::Tensor &', 'set_symint_outf', '(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out)']
at::Tensor & checkpoint_set_symint_outf(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::set_symint_outf(vec.at(0), source, storage_offset, size_, stride_, out)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset_out", rt, {self, out})[0];
}

/// ['aten::set.source_Storage_storage_offset', 'at::Tensor', 'set', '(const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride={})']
at::Tensor checkpoint_set(const at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::set(vec.at(0), source, storage_offset, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset", rt, {self})[0];
}

/// ['aten::set.source_Storage_storage_offset', 'at::Tensor', 'set_symint', '(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride={})']
at::Tensor checkpoint_set_symint(const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::set_symint(vec.at(0), source, storage_offset, size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Storage_storage_offset", rt, {self})[0];
}

/// ['aten::set.source_Tensor_out', 'at::Tensor &', 'set_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & source)']
at::Tensor & checkpoint_set_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & source) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::set_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::set.source_Tensor_out", rt, {out, self, source})[0];
}

/// ['aten::set.source_Tensor_out', 'at::Tensor &', 'set_outf', '(const at::Tensor & self, const at::Tensor & source, at::Tensor & out)']
at::Tensor & checkpoint_set_outf(const at::Tensor & self, const at::Tensor & source, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::set_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::set.source_Tensor_out", rt, {self, source, out})[0];
}

/// ['aten::set.source_Tensor', 'at::Tensor', 'set', '(const at::Tensor & self, const at::Tensor & source)']
at::Tensor checkpoint_set(const at::Tensor & self, const at::Tensor & source) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::set(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::set.source_Tensor", rt, {self, source})[0];
}

/// ['aten::set.out', 'at::Tensor &', 'set_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_set_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::set_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::set.out", rt, {out, self})[0];
}

/// ['aten::set.out', 'at::Tensor &', 'set_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_set_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::set_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::set.out", rt, {self, out})[0];
}

/// ['aten::set', 'at::Tensor', 'set', '(const at::Tensor & self)']
at::Tensor checkpoint_set(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::set(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::set", rt, {self})[0];
}

/// ['aten::clone', 'at::Tensor', 'clone', '(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::clone(vec.at(0), memory_format)};
    };
  return CheckpointTensorImpl::make("aten::clone", rt, {self})[0];
}

/// ['aten::clone.out', 'at::Tensor &', 'clone_out', '(at::Tensor & out, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor & checkpoint_clone_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::clone_out(out, vec.at(1), memory_format)};
    };
  return CheckpointTensorImpl::make("aten::clone.out", rt, {out, self})[0];
}

/// ['aten::clone.out', 'at::Tensor &', 'clone_outf', '(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out)']
at::Tensor & checkpoint_clone_outf(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::clone_outf(vec.at(0), memory_format, out)};
    };
  return CheckpointTensorImpl::make("aten::clone.out", rt, {self, out})[0];
}

/// ['aten::topk.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'topk_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_topk_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::topk_out(values, indices, vec.at(2), k, dim, largest, sorted);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk.values", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::topk.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'topk_outf', '(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_topk_outf(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::topk_outf(vec.at(0), k, dim, largest, sorted, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk.values", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::topk.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'topk_symint_out', '(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::SymInt k, int64_t dim=-1, bool largest=true, bool sorted=true)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_topk_symint_out(at::Tensor & values, at::Tensor & indices, const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(0);
      Tensor indices = vec.at(1);
      auto ret = at::topk_symint_out(values, indices, vec.at(2), k, dim, largest, sorted);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk.values", rt, {values, indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::topk.values', '::std::tuple<at::Tensor &,at::Tensor &>', 'topk_symint_outf', '(const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_topk_symint_outf(const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor values = vec.at(1);
      Tensor indices = vec.at(2);
      auto ret = at::topk_symint_outf(vec.at(0), k, dim, largest, sorted, values, indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk.values", rt, {self, values, indices});
  return {ret[0], ret[1]};
}

/// ['aten::topk', '::std::tuple<at::Tensor,at::Tensor>', 'topk', '(const at::Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_topk(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::topk(vec.at(0), k, dim, largest, sorted);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::topk', '::std::tuple<at::Tensor,at::Tensor>', 'topk_symint', '(const at::Tensor & self, c10::SymInt k, int64_t dim=-1, bool largest=true, bool sorted=true)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_topk_symint(const at::Tensor & self, c10::SymInt k, int64_t dim, bool largest, bool sorted) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::topk_symint(vec.at(0), k, dim, largest, sorted);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::topk", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::sum', 'at::Tensor', 'sum', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_sum(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum", rt, {self})[0];
}

/// ['aten::sum.dim_IntList', 'at::Tensor', 'sum', '(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_sum(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      if(dim.has_value())
        return {at::sum(vec.at(0), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype)};
      else
        return {at::sum(vec.at(0), dim_, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum.dim_IntList", rt, {self})[0];
}

/// ['aten::sum.dim_DimnameList', 'at::Tensor', 'sum', '(const at::Tensor & self, at::DimnameList dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_sum(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sum(vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum.dim_DimnameList", rt, {self})[0];
}

/// ['aten::sum.IntList_out', 'at::Tensor &', 'sum_out', '(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_sum_out(at::Tensor & out, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      if(dim.has_value())
        return {at::sum_out(out, vec.at(1), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype)};
      else
        return {at::sum_out(out, vec.at(1), dim_, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum.IntList_out", rt, {out, self})[0];
}

/// ['aten::sum.IntList_out', 'at::Tensor &', 'sum_outf', '(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_sum_outf(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  std::vector<int64_t> dim_; 
  if(dim.has_value()){
    dim_ = dim.value().vec();
  }
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      if(dim.has_value())
        return {at::sum_outf(vec.at(0), at::OptionalIntArrayRef(ArrayRef<int64_t>(dim_)), keepdim, dtype, out)};
      else
        return {at::sum_outf(vec.at(0), dim_, keepdim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::sum.IntList_out", rt, {self, out})[0];
}

/// ['aten::sum.DimnameList_out', 'at::Tensor &', 'sum_out', '(at::Tensor & out, const at::Tensor & self, at::DimnameList dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_sum_out(at::Tensor & out, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::sum_out(out, vec.at(1), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum.DimnameList_out", rt, {out, self})[0];
}

/// ['aten::sum.DimnameList_out', 'at::Tensor &', 'sum_outf', '(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_sum_outf(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::sum_outf(vec.at(0), dim, keepdim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::sum.DimnameList_out", rt, {self, out})[0];
}

/// ['aten::sum.out', 'at::Tensor &', 'sum_out', '(at::Tensor & out, const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_sum_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::sum_out(out, vec.at(1), dtype)};
    };
  return CheckpointTensorImpl::make("aten::sum.out", rt, {out, self})[0];
}

/// ['aten::sum.out', 'at::Tensor &', 'sum_outf', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_sum_outf(const at::Tensor & self, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::sum_outf(vec.at(0), dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::sum.out", rt, {self, out})[0];
}

/// ['aten::masked_fill.Scalar', 'at::Tensor', 'masked_fill', '(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value)']
at::Tensor checkpoint_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::masked_fill(vec.at(0), vec.at(1), value)};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Scalar", rt, {self, mask})[0];
}

/// ['aten::masked_fill.Tensor', 'at::Tensor', 'masked_fill', '(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value)']
at::Tensor checkpoint_masked_fill(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::masked_fill(vec.at(0), vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Tensor", rt, {self, mask, value})[0];
}

/// ['aten::masked_fill.Scalar_out', 'at::Tensor &', 'masked_fill_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value)']
at::Tensor & checkpoint_masked_fill_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::masked_fill_out(out, vec.at(1), vec.at(2), value)};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Scalar_out", rt, {out, self, mask})[0];
}

/// ['aten::masked_fill.Scalar_out', 'at::Tensor &', 'masked_fill_outf', '(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value, at::Tensor & out)']
at::Tensor & checkpoint_masked_fill_outf(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::masked_fill_outf(vec.at(0), vec.at(1), value, out)};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Scalar_out", rt, {self, mask, out})[0];
}

/// ['aten::masked_fill.Tensor_out', 'at::Tensor &', 'masked_fill_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value)']
at::Tensor & checkpoint_masked_fill_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::masked_fill_out(out, vec.at(1), vec.at(2), vec.at(3))};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Tensor_out", rt, {out, self, mask, value})[0];
}

/// ['aten::masked_fill.Tensor_out', 'at::Tensor &', 'masked_fill_outf', '(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value, at::Tensor & out)']
at::Tensor & checkpoint_masked_fill_outf(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(3);
      return {at::masked_fill_outf(vec.at(0), vec.at(1), vec.at(2), out)};
    };
  return CheckpointTensorImpl::make("aten::masked_fill.Tensor_out", rt, {self, mask, value, out})[0];
}

/// ['aten::max.dim', '::std::tuple<at::Tensor,at::Tensor>', 'max', '(const at::Tensor & self, int64_t dim, bool keepdim=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_max(const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max(vec.at(0), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.dim", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::max.dim_max', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_out', '(at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, int64_t dim, bool keepdim=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_out(at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor max = vec.at(0);
      Tensor max_values = vec.at(1);
      auto ret = at::max_out(max, max_values, vec.at(2), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.dim_max", rt, {max, max_values, self});
  return {ret[0], ret[1]};
}

/// ['aten::max.dim_max', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_outf', '(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor max = vec.at(1);
      Tensor max_values = vec.at(2);
      auto ret = at::max_outf(vec.at(0), dim, keepdim, max, max_values);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.dim_max", rt, {self, max, max_values});
  return {ret[0], ret[1]};
}

/// ['aten::max.names_dim', '::std::tuple<at::Tensor,at::Tensor>', 'max', '(const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_max(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::max(vec.at(0), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.names_dim", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::max.names_dim_max', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_out', '(at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_out(at::Tensor & max, at::Tensor & max_values, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor max = vec.at(0);
      Tensor max_values = vec.at(1);
      auto ret = at::max_out(max, max_values, vec.at(2), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.names_dim_max", rt, {max, max_values, self});
  return {ret[0], ret[1]};
}

/// ['aten::max.names_dim_max', '::std::tuple<at::Tensor &,at::Tensor &>', 'max_outf', '(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_max_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor max = vec.at(1);
      Tensor max_values = vec.at(2);
      auto ret = at::max_outf(vec.at(0), dim, keepdim, max, max_values);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::max.names_dim_max", rt, {self, max, max_values});
  return {ret[0], ret[1]};
}

/// ['aten::max', 'at::Tensor', 'max', '(const at::Tensor & self)']
at::Tensor checkpoint_max(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::max", rt, {self})[0];
}

/// ['aten::max.other', 'at::Tensor', 'max', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_max(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::max(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::max.other", rt, {self, other})[0];
}

/// ['aten::max.out', 'at::Tensor &', 'max_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_max_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::max_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::max.out", rt, {out, self, other})[0];
}

/// ['aten::max.out', 'at::Tensor &', 'max_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_max_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::max_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::max.out", rt, {self, other, out})[0];
}

/// ['aten::max.unary_out', 'at::Tensor &', 'max_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_max_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::max_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::max.unary_out", rt, {out, self})[0];
}

/// ['aten::max.unary_out', 'at::Tensor &', 'max_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_max_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::max_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::max.unary_out", rt, {self, out})[0];
}

/// ['aten::scatter.src', 'at::Tensor', 'scatter', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src)']
at::Tensor checkpoint_scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::scatter.src", rt, {self, index, src})[0];
}

/// ['aten::scatter.src_out', 'at::Tensor &', 'scatter_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src)']
at::Tensor & checkpoint_scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::scatter_out(out, vec.at(1), dim, vec.at(2), vec.at(3))};
    };
  return CheckpointTensorImpl::make("aten::scatter.src_out", rt, {out, self, index, src})[0];
}

/// ['aten::scatter.src_out', 'at::Tensor &', 'scatter_outf', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out)']
at::Tensor & checkpoint_scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(3);
      return {at::scatter_outf(vec.at(0), dim, vec.at(1), vec.at(2), out)};
    };
  return CheckpointTensorImpl::make("aten::scatter.src_out", rt, {self, index, src, out})[0];
}

/// ['aten::scatter.value', 'at::Tensor', 'scatter', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value)']
at::Tensor checkpoint_scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), value)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value", rt, {self, index})[0];
}

/// ['aten::scatter.value_out', 'at::Tensor &', 'scatter_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value)']
at::Tensor & checkpoint_scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::scatter_out(out, vec.at(1), dim, vec.at(2), value)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value_out", rt, {out, self, index})[0];
}

/// ['aten::scatter.value_out', 'at::Tensor &', 'scatter_outf', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out)']
at::Tensor & checkpoint_scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::scatter_outf(vec.at(0), dim, vec.at(1), value, out)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value_out", rt, {self, index, out})[0];
}

/// ['aten::scatter.reduce', 'at::Tensor', 'scatter', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce)']
at::Tensor checkpoint_scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), vec.at(2), reduce)};
    };
  return CheckpointTensorImpl::make("aten::scatter.reduce", rt, {self, index, src})[0];
}

/// ['aten::scatter.reduce_out', 'at::Tensor &', 'scatter_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce)']
at::Tensor & checkpoint_scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::scatter_out(out, vec.at(1), dim, vec.at(2), vec.at(3), reduce)};
    };
  return CheckpointTensorImpl::make("aten::scatter.reduce_out", rt, {out, self, index, src})[0];
}

/// ['aten::scatter.reduce_out', 'at::Tensor &', 'scatter_outf', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out)']
at::Tensor & checkpoint_scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(3);
      return {at::scatter_outf(vec.at(0), dim, vec.at(1), vec.at(2), reduce, out)};
    };
  return CheckpointTensorImpl::make("aten::scatter.reduce_out", rt, {self, index, src, out})[0];
}

/// ['aten::scatter.value_reduce', 'at::Tensor', 'scatter', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce)']
at::Tensor checkpoint_scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), value, reduce)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value_reduce", rt, {self, index})[0];
}

/// ['aten::scatter.value_reduce_out', 'at::Tensor &', 'scatter_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce)']
at::Tensor & checkpoint_scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::scatter_out(out, vec.at(1), dim, vec.at(2), value, reduce)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value_reduce_out", rt, {out, self, index})[0];
}

/// ['aten::scatter.value_reduce_out', 'at::Tensor &', 'scatter_outf', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out)']
at::Tensor & checkpoint_scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::scatter_outf(vec.at(0), dim, vec.at(1), value, reduce, out)};
    };
  return CheckpointTensorImpl::make("aten::scatter.value_reduce_out", rt, {self, index, out})[0];
}

/// ['aten::scatter.dimname_src', 'at::Tensor', 'scatter', '(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src)']
at::Tensor checkpoint_scatter(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::scatter.dimname_src", rt, {self, index, src})[0];
}

/// ['aten::scatter.dimname_value', 'at::Tensor', 'scatter', '(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value)']
at::Tensor checkpoint_scatter(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::scatter(vec.at(0), dim, vec.at(1), value)};
    };
  return CheckpointTensorImpl::make("aten::scatter.dimname_value", rt, {self, index})[0];
}

/// ['aten::eq.Scalar_out', 'at::Tensor &', 'eq_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_eq_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::eq_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::eq.Scalar_out", rt, {out, self})[0];
}

/// ['aten::eq.Scalar_out', 'at::Tensor &', 'eq_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_eq_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::eq_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::eq.Scalar_out", rt, {self, out})[0];
}

/// ['aten::eq.Scalar', 'at::Tensor', 'eq', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_eq(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::eq.Scalar", rt, {self})[0];
}

/// ['aten::eq.Tensor_out', 'at::Tensor &', 'eq_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_eq_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::eq_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::eq.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::eq.Tensor_out', 'at::Tensor &', 'eq_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_eq_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::eq_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::eq.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::eq.Tensor', 'at::Tensor', 'eq', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_eq(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::eq(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::eq.Tensor", rt, {self, other})[0];
}

/// ['aten::prod', 'at::Tensor', 'prod', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_prod(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::prod(vec.at(0), dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod", rt, {self})[0];
}

/// ['aten::prod.dim_int', 'at::Tensor', 'prod', '(const at::Tensor & self, int64_t dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_prod(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::prod(vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod.dim_int", rt, {self})[0];
}

/// ['aten::prod.int_out', 'at::Tensor &', 'prod_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_prod_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::prod_out(out, vec.at(1), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod.int_out", rt, {out, self})[0];
}

/// ['aten::prod.int_out', 'at::Tensor &', 'prod_outf', '(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_prod_outf(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::prod_outf(vec.at(0), dim, keepdim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::prod.int_out", rt, {self, out})[0];
}

/// ['aten::prod.dim_Dimname', 'at::Tensor', 'prod', '(const at::Tensor & self, at::Dimname dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_prod(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::prod(vec.at(0), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod.dim_Dimname", rt, {self})[0];
}

/// ['aten::prod.Dimname_out', 'at::Tensor &', 'prod_out', '(at::Tensor & out, const at::Tensor & self, at::Dimname dim, bool keepdim=false, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_prod_out(at::Tensor & out, const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::prod_out(out, vec.at(1), dim, keepdim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod.Dimname_out", rt, {out, self})[0];
}

/// ['aten::prod.Dimname_out', 'at::Tensor &', 'prod_outf', '(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_prod_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::prod_outf(vec.at(0), dim, keepdim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::prod.Dimname_out", rt, {self, out})[0];
}

/// ['aten::prod.out', 'at::Tensor &', 'prod_out', '(at::Tensor & out, const at::Tensor & self, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_prod_out(at::Tensor & out, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::prod_out(out, vec.at(1), dtype)};
    };
  return CheckpointTensorImpl::make("aten::prod.out", rt, {out, self})[0];
}

/// ['aten::prod.out', 'at::Tensor &', 'prod_outf', '(const at::Tensor & self, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_prod_outf(const at::Tensor & self, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::prod_outf(vec.at(0), dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::prod.out", rt, {self, out})[0];
}

/// ['aten::cumsum', 'at::Tensor', 'cumsum', '(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_cumsum(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cumsum(vec.at(0), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::cumsum", rt, {self})[0];
}

/// ['aten::cumsum.out', 'at::Tensor &', 'cumsum_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_cumsum_out(at::Tensor & out, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::cumsum_out(out, vec.at(1), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::cumsum.out", rt, {out, self})[0];
}

/// ['aten::cumsum.out', 'at::Tensor &', 'cumsum_outf', '(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_cumsum_outf(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::cumsum_outf(vec.at(0), dim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::cumsum.out", rt, {self, out})[0];
}

/// ['aten::cumsum.dimname', 'at::Tensor', 'cumsum', '(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_cumsum(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::cumsum(vec.at(0), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::cumsum.dimname", rt, {self})[0];
}

/// ['aten::cumsum.dimname_out', 'at::Tensor &', 'cumsum_out', '(at::Tensor & out, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_cumsum_out(at::Tensor & out, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::cumsum_out(out, vec.at(1), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::cumsum.dimname_out", rt, {out, self})[0];
}

/// ['aten::cumsum.dimname_out', 'at::Tensor &', 'cumsum_outf', '(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_cumsum_outf(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::cumsum_outf(vec.at(0), dim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::cumsum.dimname_out", rt, {self, out})[0];
}

/// ['aten::embedding', 'at::Tensor', 'embedding', '(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false)']
at::Tensor checkpoint_embedding(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding(vec.at(0), vec.at(1), padding_idx, scale_grad_by_freq, sparse)};
    };
  return CheckpointTensorImpl::make("aten::embedding", rt, {weight, indices})[0];
}

/// ['aten::embedding', 'at::Tensor', 'embedding_symint', '(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false)']
at::Tensor checkpoint_embedding_symint(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::embedding_symint(vec.at(0), vec.at(1), padding_idx, scale_grad_by_freq, sparse)};
    };
  return CheckpointTensorImpl::make("aten::embedding", rt, {weight, indices})[0];
}

/// ['aten::embedding.out', 'at::Tensor &', 'embedding_out', '(at::Tensor & out, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false)']
at::Tensor & checkpoint_embedding_out(at::Tensor & out, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::embedding_out(out, vec.at(1), vec.at(2), padding_idx, scale_grad_by_freq, sparse)};
    };
  return CheckpointTensorImpl::make("aten::embedding.out", rt, {out, weight, indices})[0];
}

/// ['aten::embedding.out', 'at::Tensor &', 'embedding_outf', '(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out)']
at::Tensor & checkpoint_embedding_outf(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::embedding_outf(vec.at(0), vec.at(1), padding_idx, scale_grad_by_freq, sparse, out)};
    };
  return CheckpointTensorImpl::make("aten::embedding.out", rt, {weight, indices, out})[0];
}

/// ['aten::embedding.out', 'at::Tensor &', 'embedding_symint_out', '(at::Tensor & out, const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false)']
at::Tensor & checkpoint_embedding_symint_out(at::Tensor & out, const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::embedding_symint_out(out, vec.at(1), vec.at(2), padding_idx, scale_grad_by_freq, sparse)};
    };
  return CheckpointTensorImpl::make("aten::embedding.out", rt, {out, weight, indices})[0];
}

/// ['aten::embedding.out', 'at::Tensor &', 'embedding_symint_outf', '(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out)']
at::Tensor & checkpoint_embedding_symint_outf(const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::embedding_symint_outf(vec.at(0), vec.at(1), padding_idx, scale_grad_by_freq, sparse, out)};
    };
  return CheckpointTensorImpl::make("aten::embedding.out", rt, {weight, indices, out})[0];
}

/// ['aten::min.dim', '::std::tuple<at::Tensor,at::Tensor>', 'min', '(const at::Tensor & self, int64_t dim, bool keepdim=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_min(const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::min(vec.at(0), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.dim", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::min.dim_min', '::std::tuple<at::Tensor &,at::Tensor &>', 'min_out', '(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, int64_t dim, bool keepdim=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_min_out(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor min = vec.at(0);
      Tensor min_indices = vec.at(1);
      auto ret = at::min_out(min, min_indices, vec.at(2), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.dim_min", rt, {min, min_indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::min.dim_min', '::std::tuple<at::Tensor &,at::Tensor &>', 'min_outf', '(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_min_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor min = vec.at(1);
      Tensor min_indices = vec.at(2);
      auto ret = at::min_outf(vec.at(0), dim, keepdim, min, min_indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.dim_min", rt, {self, min, min_indices});
  return {ret[0], ret[1]};
}

/// ['aten::min.names_dim', '::std::tuple<at::Tensor,at::Tensor>', 'min', '(const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
::std::tuple<at::Tensor,at::Tensor> checkpoint_min(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      auto ret = at::min(vec.at(0), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.names_dim", rt, {self});
  return {ret[0], ret[1]};
}

/// ['aten::min.names_dim_min', '::std::tuple<at::Tensor &,at::Tensor &>', 'min_out', '(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_min_out(at::Tensor & min, at::Tensor & min_indices, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor min = vec.at(0);
      Tensor min_indices = vec.at(1);
      auto ret = at::min_out(min, min_indices, vec.at(2), dim, keepdim);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.names_dim_min", rt, {min, min_indices, self});
  return {ret[0], ret[1]};
}

/// ['aten::min.names_dim_min', '::std::tuple<at::Tensor &,at::Tensor &>', 'min_outf', '(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices)']
::std::tuple<at::Tensor &,at::Tensor &> checkpoint_min_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor min = vec.at(1);
      Tensor min_indices = vec.at(2);
      auto ret = at::min_outf(vec.at(0), dim, keepdim, min, min_indices);
      return {std::get<0>(ret), std::get<1>(ret)};
    };
  auto ret = CheckpointTensorImpl::make("aten::min.names_dim_min", rt, {self, min, min_indices});
  return {ret[0], ret[1]};
}

/// ['aten::min', 'at::Tensor', 'min', '(const at::Tensor & self)']
at::Tensor checkpoint_min(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::min(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::min", rt, {self})[0];
}

/// ['aten::min.unary_out', 'at::Tensor &', 'min_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_min_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::min_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::min.unary_out", rt, {out, self})[0];
}

/// ['aten::min.unary_out', 'at::Tensor &', 'min_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_min_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::min_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::min.unary_out", rt, {self, out})[0];
}

/// ['aten::min.out', 'at::Tensor &', 'min_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_min_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::min_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::min.out", rt, {out, self, other})[0];
}

/// ['aten::min.out', 'at::Tensor &', 'min_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_min_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::min_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::min.out", rt, {self, other, out})[0];
}

/// ['aten::min.other', 'at::Tensor', 'min', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_min(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::min(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::min.other", rt, {self, other})[0];
}

/// ['aten::sub.out', 'at::Tensor &', 'sub_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_sub_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::sub_out(out, vec.at(1), vec.at(2), alpha)};
    };
  return CheckpointTensorImpl::make("aten::sub.out", rt, {out, self, other})[0];
}

/// ['aten::sub.out', 'at::Tensor &', 'sub_outf', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_sub_outf(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::sub_outf(vec.at(0), vec.at(1), alpha, out)};
    };
  return CheckpointTensorImpl::make("aten::sub.out", rt, {self, other, out})[0];
}

/// ['aten::sub.Tensor', 'at::Tensor', 'sub', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_sub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec.at(0), vec.at(1), alpha)};
    };
  return CheckpointTensorImpl::make("aten::sub.Tensor", rt, {self, other})[0];
}

/// ['aten::sub.Scalar', 'at::Tensor', 'sub', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_sub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::sub(vec.at(0), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::sub.Scalar", rt, {self})[0];
}

/// ['aten::sub.Scalar_out', 'at::Tensor &', 'sub_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_sub_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::sub_out(out, vec.at(1), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::sub.Scalar_out", rt, {out, self})[0];
}

/// ['aten::sub.Scalar_out', 'at::Tensor &', 'sub_outf', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_sub_outf(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::sub_outf(vec.at(0), other, alpha, out)};
    };
  return CheckpointTensorImpl::make("aten::sub.Scalar_out", rt, {self, out})[0];
}

/// ['aten::fill.Scalar', 'at::Tensor', 'fill', '(const at::Tensor & self, const at::Scalar & value)']
at::Tensor checkpoint_fill(const at::Tensor & self, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::fill(vec.at(0), value)};
    };
  return CheckpointTensorImpl::make("aten::fill.Scalar", rt, {self})[0];
}

/// ['aten::fill.Tensor', 'at::Tensor', 'fill', '(const at::Tensor & self, const at::Tensor & value)']
at::Tensor checkpoint_fill(const at::Tensor & self, const at::Tensor & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::fill(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::fill.Tensor", rt, {self, value})[0];
}

/// ['aten::fill_.Scalar', 'at::Tensor &', 'fill_', '(at::Tensor & self, const at::Scalar & value)']
// at::Tensor & checkpoint_fill_(at::Tensor & self, const at::Scalar & value) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor self = vec.at(0);
//       return {at::fill_(self, value)};
//     };
//   return CheckpointTensorImpl::make("aten::fill_.Scalar", rt, {self})[0];
// }

// /// ['aten::fill_.Tensor', 'at::Tensor &', 'fill_', '(at::Tensor & self, const at::Tensor & value)']
// at::Tensor & checkpoint_fill_(at::Tensor & self, const at::Tensor & value) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       Tensor self = vec.at(0);
//       return {at::fill_(self, vec.at(1))};
//     };
//   return CheckpointTensorImpl::make("aten::fill_.Tensor", rt, {self, value})[0];
// }

/// ['aten::fill.Scalar_out', 'at::Tensor &', 'fill_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & value)']
at::Tensor & checkpoint_fill_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::fill_out(out, vec.at(1), value)};
    };
  return CheckpointTensorImpl::make("aten::fill.Scalar_out", rt, {out, self})[0];
}

/// ['aten::fill.Scalar_out', 'at::Tensor &', 'fill_outf', '(const at::Tensor & self, const at::Scalar & value, at::Tensor & out)']
at::Tensor & checkpoint_fill_outf(const at::Tensor & self, const at::Scalar & value, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::fill_outf(vec.at(0), value, out)};
    };
  return CheckpointTensorImpl::make("aten::fill.Scalar_out", rt, {self, out})[0];
}

/// ['aten::fill.Tensor_out', 'at::Tensor &', 'fill_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & value)']
at::Tensor & checkpoint_fill_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & value) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::fill_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::fill.Tensor_out", rt, {out, self, value})[0];
}

/// ['aten::fill.Tensor_out', 'at::Tensor &', 'fill_outf', '(const at::Tensor & self, const at::Tensor & value, at::Tensor & out)']
at::Tensor & checkpoint_fill_outf(const at::Tensor & self, const at::Tensor & value, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::fill_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::fill.Tensor_out", rt, {self, value, out})[0];
}

/// ['aten::index_select.out', 'at::Tensor &', 'index_select_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index)']
at::Tensor & checkpoint_index_select_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::index_select_out(out, vec.at(1), dim, vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::index_select.out", rt, {out, self, index})[0];
}

/// ['aten::index_select.out', 'at::Tensor &', 'index_select_outf', '(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out)']
at::Tensor & checkpoint_index_select_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::index_select_outf(vec.at(0), dim, vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::index_select.out", rt, {self, index, out})[0];
}

/// ['aten::index_select', 'at::Tensor', 'index_select', '(const at::Tensor & self, int64_t dim, const at::Tensor & index)']
at::Tensor checkpoint_index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::index_select(vec.at(0), dim, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::index_select", rt, {self, index})[0];
}

/// ['aten::index_select.dimname_out', 'at::Tensor &', 'index_select_out', '(at::Tensor & out, const at::Tensor & self, at::Dimname dim, const at::Tensor & index)']
at::Tensor & checkpoint_index_select_out(at::Tensor & out, const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::index_select_out(out, vec.at(1), dim, vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::index_select.dimname_out", rt, {out, self, index})[0];
}

/// ['aten::index_select.dimname_out', 'at::Tensor &', 'index_select_outf', '(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out)']
at::Tensor & checkpoint_index_select_outf(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::index_select_outf(vec.at(0), dim, vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::index_select.dimname_out", rt, {self, index, out})[0];
}

/// ['aten::index_select.dimname', 'at::Tensor', 'index_select', '(const at::Tensor & self, at::Dimname dim, const at::Tensor & index)']
at::Tensor checkpoint_index_select(const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::index_select(vec.at(0), dim, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::index_select.dimname", rt, {self, index})[0];
}

/// ['aten::rsub.Tensor', 'at::Tensor', 'rsub', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_rsub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), vec.at(1), alpha)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Tensor", rt, {self, other})[0];
}

/// ['aten::rsub.Scalar', 'at::Tensor', 'rsub', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor checkpoint_rsub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::rsub(vec.at(0), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Scalar", rt, {self})[0];
}

/// ['aten::rsub.Tensor_out', 'at::Tensor &', 'rsub_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_rsub_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::rsub_out(out, vec.at(1), vec.at(2), alpha)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::rsub.Tensor_out', 'at::Tensor &', 'rsub_outf', '(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_rsub_outf(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::rsub_outf(vec.at(0), vec.at(1), alpha, out)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::rsub.Scalar_out', 'at::Tensor &', 'rsub_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha=1)']
at::Tensor & checkpoint_rsub_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::rsub_out(out, vec.at(1), other, alpha)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Scalar_out", rt, {out, self})[0];
}

/// ['aten::rsub.Scalar_out', 'at::Tensor &', 'rsub_outf', '(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out)']
at::Tensor & checkpoint_rsub_outf(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::rsub_outf(vec.at(0), other, alpha, out)};
    };
  return CheckpointTensorImpl::make("aten::rsub.Scalar_out", rt, {self, out})[0];
}

/// ['aten::div.Tensor', 'at::Tensor', 'div', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_div(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::div.Tensor", rt, {self, other})[0];
}

/// ['aten::div.out', 'at::Tensor &', 'div_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_div_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::div_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::div.out", rt, {out, self, other})[0];
}

/// ['aten::div.out', 'at::Tensor &', 'div_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_div_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::div_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::div.out", rt, {self, other, out})[0];
}

/// ['aten::div.Tensor_mode', 'at::Tensor', 'div', '(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode)']
at::Tensor checkpoint_div(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), vec.at(1), rounding_mode)};
    };
  return CheckpointTensorImpl::make("aten::div.Tensor_mode", rt, {self, other})[0];
}

/// ['aten::div.out_mode', 'at::Tensor &', 'div_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode)']
at::Tensor & checkpoint_div_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::div_out(out, vec.at(1), vec.at(2), rounding_mode)};
    };
  return CheckpointTensorImpl::make("aten::div.out_mode", rt, {out, self, other})[0];
}

/// ['aten::div.out_mode', 'at::Tensor &', 'div_outf', '(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out)']
at::Tensor & checkpoint_div_outf(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::div_outf(vec.at(0), vec.at(1), rounding_mode, out)};
    };
  return CheckpointTensorImpl::make("aten::div.out_mode", rt, {self, other, out})[0];
}

/// ['aten::div.Scalar', 'at::Tensor', 'div', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_div(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar", rt, {self})[0];
}

/// ['aten::div.Scalar_mode', 'at::Tensor', 'div', '(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode)']
at::Tensor checkpoint_div(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::div(vec.at(0), other, rounding_mode)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar_mode", rt, {self})[0];
}

/// ['aten::div.Scalar_out', 'at::Tensor &', 'div_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_div_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::div_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar_out", rt, {out, self})[0];
}

/// ['aten::div.Scalar_out', 'at::Tensor &', 'div_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_div_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::div_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar_out", rt, {self, out})[0];
}

/// ['aten::div.Scalar_mode_out', 'at::Tensor &', 'div_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode)']
at::Tensor & checkpoint_div_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::div_out(out, vec.at(1), other, rounding_mode)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar_mode_out", rt, {out, self})[0];
}

/// ['aten::div.Scalar_mode_out', 'at::Tensor &', 'div_outf', '(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out)']
at::Tensor & checkpoint_div_outf(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::div_outf(vec.at(0), other, rounding_mode, out)};
    };
  return CheckpointTensorImpl::make("aten::div.Scalar_mode_out", rt, {self, out})[0];
}

/// ['aten::arange', 'at::Tensor', 'arange', '(const at::Scalar & end, at::TensorOptions options={})']
at::Tensor checkpoint_arange(const at::Scalar & end, at::TensorOptions options) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(end, options)};
    };
  return CheckpointTensorImpl::make("aten::arange", rt, {})[0];
}

/// ['aten::arange', 'at::Tensor', 'arange', '(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_arange(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(end, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::arange", rt, {})[0];
}

/// ['aten::arange.start', 'at::Tensor', 'arange', '(const at::Scalar & start, const at::Scalar & end, at::TensorOptions options={})']
at::Tensor checkpoint_arange(const at::Scalar & start, const at::Scalar & end, at::TensorOptions options) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(start, end, options)};
    };
  return CheckpointTensorImpl::make("aten::arange.start", rt, {})[0];
}

/// ['aten::arange.start', 'at::Tensor', 'arange', '(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_arange(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(start, end, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::arange.start", rt, {})[0];
}

/// ['aten::arange.start_step', 'at::Tensor', 'arange', '(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::TensorOptions options={})']
at::Tensor checkpoint_arange(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::TensorOptions options) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(start, end, step, options)};
    };
  return CheckpointTensorImpl::make("aten::arange.start_step", rt, {})[0];
}

/// ['aten::arange.start_step', 'at::Tensor', 'arange', '(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_arange(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::arange(start, end, step, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::arange.start_step", rt, {})[0];
}

/// ['aten::arange.out', 'at::Tensor &', 'arange_out', '(at::Tensor & out, const at::Scalar & end)']
at::Tensor & checkpoint_arange_out(at::Tensor & out, const at::Scalar & end) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::arange_out(out, end)};
    };
  return CheckpointTensorImpl::make("aten::arange.out", rt, {out})[0];
}

/// ['aten::arange.out', 'at::Tensor &', 'arange_outf', '(const at::Scalar & end, at::Tensor & out)']
at::Tensor & checkpoint_arange_outf(const at::Scalar & end, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::arange_outf(end, out)};
    };
  return CheckpointTensorImpl::make("aten::arange.out", rt, {out})[0];
}

/// ['aten::arange.start_out', 'at::Tensor &', 'arange_out', '(at::Tensor & out, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step)']
at::Tensor & checkpoint_arange_out(at::Tensor & out, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::arange_out(out, start, end, step)};
    };
  return CheckpointTensorImpl::make("aten::arange.start_out", rt, {out})[0];
}

/// ['aten::arange.start_out', 'at::Tensor &', 'arange_outf', '(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out)']
at::Tensor & checkpoint_arange_outf(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::arange_outf(start, end, step, out)};
    };
  return CheckpointTensorImpl::make("aten::arange.start_out", rt, {out})[0];
}

/// ['aten::any.dim', 'at::Tensor', 'any', '(const at::Tensor & self, int64_t dim, bool keepdim=false)']
at::Tensor checkpoint_any(const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::any(vec.at(0), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::any.dim", rt, {self})[0];
}

/// ['aten::any.out', 'at::Tensor &', 'any_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim=false)']
at::Tensor & checkpoint_any_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::any_out(out, vec.at(1), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::any.out", rt, {out, self})[0];
}

/// ['aten::any.out', 'at::Tensor &', 'any_outf', '(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out)']
at::Tensor & checkpoint_any_outf(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::any_outf(vec.at(0), dim, keepdim, out)};
    };
  return CheckpointTensorImpl::make("aten::any.out", rt, {self, out})[0];
}

/// ['aten::any.dimname', 'at::Tensor', 'any', '(const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
at::Tensor checkpoint_any(const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::any(vec.at(0), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::any.dimname", rt, {self})[0];
}

/// ['aten::any.dimname_out', 'at::Tensor &', 'any_out', '(at::Tensor & out, const at::Tensor & self, at::Dimname dim, bool keepdim=false)']
at::Tensor & checkpoint_any_out(at::Tensor & out, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::any_out(out, vec.at(1), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::any.dimname_out", rt, {out, self})[0];
}

/// ['aten::any.dimname_out', 'at::Tensor &', 'any_outf', '(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out)']
at::Tensor & checkpoint_any_outf(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::any_outf(vec.at(0), dim, keepdim, out)};
    };
  return CheckpointTensorImpl::make("aten::any.dimname_out", rt, {self, out})[0];
}

/// ['aten::any', 'at::Tensor', 'any', '(const at::Tensor & self)']
at::Tensor checkpoint_any(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::any(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::any", rt, {self})[0];
}

/// ['aten::any.all_out', 'at::Tensor &', 'any_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_any_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::any_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::any.all_out", rt, {out, self})[0];
}

/// ['aten::any.all_out', 'at::Tensor &', 'any_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_any_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::any_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::any.all_out", rt, {self, out})[0];
}


/// ['aten::softmax.int', 'at::Tensor', 'softmax', '(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_softmax(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::softmax(vec.at(0), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::softmax.int", rt, {self})[0];
}

/// ['aten::softmax.int_out', 'at::Tensor &', 'softmax_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor & checkpoint_softmax_out(at::Tensor & out, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::softmax_out(out, vec.at(1), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::softmax.int_out", rt, {out, self})[0];
}

/// ['aten::softmax.int_out', 'at::Tensor &', 'softmax_outf', '(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out)']
at::Tensor & checkpoint_softmax_outf(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::softmax_outf(vec.at(0), dim, dtype, out)};
    };
  return CheckpointTensorImpl::make("aten::softmax.int_out", rt, {self, out})[0];
}

/// ['aten::softmax.Dimname', 'at::Tensor', 'softmax', '(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype=c10::nullopt)']
at::Tensor checkpoint_softmax(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::softmax(vec.at(0), dim, dtype)};
    };
  return CheckpointTensorImpl::make("aten::softmax.Dimname", rt, {self})[0];
}

/// ['aten::new_ones.out', 'at::Tensor &', 'new_ones_out', '(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size)']
at::Tensor & checkpoint_new_ones_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::new_ones_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_ones.out", rt, {out, self})[0];
}

/// ['aten::new_ones.out', 'at::Tensor &', 'new_ones_outf', '(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_new_ones_outf(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::new_ones_outf(vec.at(0), size_, out)};
    };
  return CheckpointTensorImpl::make("aten::new_ones.out", rt, {self, out})[0];
}

/// ['aten::new_ones.out', 'at::Tensor &', 'new_ones_symint_out', '(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size)']
at::Tensor & checkpoint_new_ones_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::new_ones_symint_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_ones.out", rt, {out, self})[0];
}

/// ['aten::new_ones.out', 'at::Tensor &', 'new_ones_symint_outf', '(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_new_ones_symint_outf(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::new_ones_symint_outf(vec.at(0), size_, out)};
    };
  return CheckpointTensorImpl::make("aten::new_ones.out", rt, {self, out})[0];
}

/// ['aten::lt.Scalar_out', 'at::Tensor &', 'lt_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_lt_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::lt_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::lt.Scalar_out", rt, {out, self})[0];
}

/// ['aten::lt.Scalar_out', 'at::Tensor &', 'lt_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_lt_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::lt_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::lt.Scalar_out", rt, {self, out})[0];
}

/// ['aten::lt.Scalar', 'at::Tensor', 'lt', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_lt(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::lt(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::lt.Scalar", rt, {self})[0];
}

/// ['aten::lt.Tensor_out', 'at::Tensor &', 'lt_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_lt_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::lt_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::lt.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::lt.Tensor_out', 'at::Tensor &', 'lt_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_lt_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::lt_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::lt.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::lt.Tensor', 'at::Tensor', 'lt', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_lt(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::lt(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::lt.Tensor", rt, {self, other})[0];
}

/// ['aten::_softmax', 'at::Tensor', '_softmax', '(const at::Tensor & self, int64_t dim, bool half_to_float)']
at::Tensor checkpoint__softmax(const at::Tensor & self, int64_t dim, bool half_to_float) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_softmax(vec.at(0), dim, half_to_float)};
    };
  return CheckpointTensorImpl::make("aten::_softmax", rt, {self})[0];
}

/// ['aten::_softmax.out', 'at::Tensor &', '_softmax_out', '(at::Tensor & out, const at::Tensor & self, int64_t dim, bool half_to_float)']
at::Tensor & checkpoint__softmax_out(at::Tensor & out, const at::Tensor & self, int64_t dim, bool half_to_float) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::_softmax_out(out, vec.at(1), dim, half_to_float)};
    };
  return CheckpointTensorImpl::make("aten::_softmax.out", rt, {out, self})[0];
}

/// ['aten::_softmax.out', 'at::Tensor &', '_softmax_outf', '(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out)']
at::Tensor & checkpoint__softmax_outf(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::_softmax_outf(vec.at(0), dim, half_to_float, out)};
    };
  return CheckpointTensorImpl::make("aten::_softmax.out", rt, {self, out})[0];
}

/// ['aten::ones.names', 'at::Tensor', 'ones', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options={})']
at::Tensor checkpoint_ones(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones(size_, names, options)};
    };
  return CheckpointTensorImpl::make("aten::ones.names", rt, {})[0];
}

/// ['aten::ones.names', 'at::Tensor', 'ones', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_ones(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones(size_, names, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::ones.names", rt, {})[0];
}

/// ['aten::ones', 'at::Tensor', 'ones', '(at::IntArrayRef size, at::TensorOptions options={})']
at::Tensor checkpoint_ones(at::IntArrayRef size, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones(size_, options)};
    };
  return CheckpointTensorImpl::make("aten::ones", rt, {})[0];
}

/// ['aten::ones', 'at::Tensor', 'ones', '(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones(size_, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::ones", rt, {})[0];
}

/// ['aten::ones', 'at::Tensor', 'ones_symint', '(c10::SymIntArrayRef size, at::TensorOptions options={})']
at::Tensor checkpoint_ones_symint(c10::SymIntArrayRef size, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones_symint(size_, options)};
    };
  return CheckpointTensorImpl::make("aten::ones", rt, {})[0];
}

/// ['aten::ones', 'at::Tensor', 'ones_symint', '(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_ones_symint(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ones_symint(size_, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::ones", rt, {})[0];
}

/// ['aten::ones.out', 'at::Tensor &', 'ones_out', '(at::Tensor & out, at::IntArrayRef size)']
at::Tensor & checkpoint_ones_out(at::Tensor & out, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_out(out, size_)};
    };
  return CheckpointTensorImpl::make("aten::ones.out", rt, {out})[0];
}

/// ['aten::ones.out', 'at::Tensor &', 'ones_outf', '(at::IntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_ones_outf(at::IntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_outf(size_, out)};
    };
  return CheckpointTensorImpl::make("aten::ones.out", rt, {out})[0];
}

/// ['aten::ones.out', 'at::Tensor &', 'ones_symint_out', '(at::Tensor & out, c10::SymIntArrayRef size)']
at::Tensor & checkpoint_ones_symint_out(at::Tensor & out, c10::SymIntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_symint_out(out, size_)};
    };
  return CheckpointTensorImpl::make("aten::ones.out", rt, {out})[0];
}

/// ['aten::ones.out', 'at::Tensor &', 'ones_symint_outf', '(c10::SymIntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_ones_symint_outf(c10::SymIntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_symint_outf(size_, out)};
    };
  return CheckpointTensorImpl::make("aten::ones.out", rt, {out})[0];
}

/// ['aten::ones.names_out', 'at::Tensor &', 'ones_out', '(at::Tensor & out, at::IntArrayRef size, c10::optional<at::DimnameList> names)']
at::Tensor & checkpoint_ones_out(at::Tensor & out, at::IntArrayRef size, c10::optional<at::DimnameList> names) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_out(out, size_, names)};
    };
  return CheckpointTensorImpl::make("aten::ones.names_out", rt, {out})[0];
}

/// ['aten::ones.names_out', 'at::Tensor &', 'ones_outf', '(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::Tensor & out)']
at::Tensor & checkpoint_ones_outf(at::IntArrayRef size, c10::optional<at::DimnameList> names, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ones_outf(size_, names, out)};
    };
  return CheckpointTensorImpl::make("aten::ones.names_out", rt, {out})[0];
}

/// ['aten::le.Scalar_out', 'at::Tensor &', 'le_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_le_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::le_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::le.Scalar_out", rt, {out, self})[0];
}

/// ['aten::le.Scalar_out', 'at::Tensor &', 'le_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_le_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::le_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::le.Scalar_out", rt, {self, out})[0];
}

/// ['aten::le.Scalar', 'at::Tensor', 'le', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_le(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::le(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::le.Scalar", rt, {self})[0];
}

/// ['aten::le.Tensor_out', 'at::Tensor &', 'le_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_le_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::le_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::le.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::le.Tensor_out', 'at::Tensor &', 'le_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_le_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::le_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::le.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::le.Tensor', 'at::Tensor', 'le', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_le(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::le(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::le.Tensor", rt, {self, other})[0];
}

/// ['aten::ne.Scalar_out', 'at::Tensor &', 'ne_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_ne_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ne_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::ne.Scalar_out", rt, {out, self})[0];
}

/// ['aten::ne.Scalar_out', 'at::Tensor &', 'ne_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_ne_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::ne_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::ne.Scalar_out", rt, {self, out})[0];
}

/// ['aten::ne.Scalar', 'at::Tensor', 'ne', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_ne(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::ne.Scalar", rt, {self})[0];
}

/// ['aten::ne.Tensor_out', 'at::Tensor &', 'ne_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_ne_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ne_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::ne.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::ne.Tensor_out', 'at::Tensor &', 'ne_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_ne_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::ne_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::ne.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::ne.Tensor', 'at::Tensor', 'ne', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_ne(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ne(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::ne.Tensor", rt, {self, other})[0];
}

/// ['aten::ge.Scalar_out', 'at::Tensor &', 'ge_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_ge_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ge_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::ge.Scalar_out", rt, {out, self})[0];
}

/// ['aten::ge.Scalar_out', 'at::Tensor &', 'ge_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_ge_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::ge_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::ge.Scalar_out", rt, {self, out})[0];
}

/// ['aten::ge.Scalar', 'at::Tensor', 'ge', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_ge(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ge(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::ge.Scalar", rt, {self})[0];
}

/// ['aten::ge.Tensor_out', 'at::Tensor &', 'ge_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_ge_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::ge_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::ge.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::ge.Tensor_out', 'at::Tensor &', 'ge_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_ge_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::ge_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::ge.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::ge.Tensor', 'at::Tensor', 'ge', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_ge(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::ge(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::ge.Tensor", rt, {self, other})[0];
}

/// ['aten::new_empty.out', 'at::Tensor &', 'new_empty_out', '(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size)']
at::Tensor & checkpoint_new_empty_out(at::Tensor & out, const at::Tensor & self, at::IntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::new_empty_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_empty.out", rt, {out, self})[0];
}

/// ['aten::new_empty.out', 'at::Tensor &', 'new_empty_outf', '(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_new_empty_outf(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::new_empty_out(out, vec.at(0), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_empty.out", rt, {self, out})[0];
}

/// ['aten::new_empty.out', 'at::Tensor &', 'new_empty_symint_out', '(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size)']
at::Tensor & checkpoint_new_empty_symint_out(at::Tensor & out, const at::Tensor & self, c10::SymIntArrayRef size) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::new_empty_symint_out(out, vec.at(1), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_empty.out", rt, {out, self})[0];
}

/// ['aten::new_empty.out', 'at::Tensor &', 'new_empty_symint_outf', '(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out)']
at::Tensor & checkpoint_new_empty_symint_outf(const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::new_empty_symint_out(out, vec.at(0), size_)};
    };
  return CheckpointTensorImpl::make("aten::new_empty.out", rt, {self, out})[0];
}

/// ['aten::exponential.out', 'at::Tensor &', 'exponential_out', '(at::Tensor & out, const at::Tensor & self, double lambd=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor & checkpoint_exponential_out(at::Tensor & out, const at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::exponential_out(out, vec.at(1), lambd, generator)};
    };
  return CheckpointTensorImpl::make("aten::exponential.out", rt, {out, self})[0];
}

/// ['aten::exponential.out', 'at::Tensor &', 'exponential_outf', '(const at::Tensor & self, double lambd, c10::optional<at::Generator> generator, at::Tensor & out)']
at::Tensor & checkpoint_exponential_outf(const at::Tensor & self, double lambd, c10::optional<at::Generator> generator, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::exponential_outf(vec.at(0), lambd, generator, out)};
    };
  return CheckpointTensorImpl::make("aten::exponential.out", rt, {self, out})[0];
}

/// ['aten::exponential', 'at::Tensor', 'exponential', '(const at::Tensor & self, double lambd=1, c10::optional<at::Generator> generator=c10::nullopt)']
at::Tensor checkpoint_exponential(const at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::exponential(vec.at(0), lambd, generator)};
    };
  return CheckpointTensorImpl::make("aten::exponential", rt, {self})[0];
}

/// ['aten::tile', 'at::Tensor', 'tile', '(const at::Tensor & self, at::IntArrayRef dims)']
at::Tensor checkpoint_tile(const at::Tensor & self, at::IntArrayRef dims) {
  auto dims_ = dims.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tile(vec.at(0), dims_)};
    };
  return CheckpointTensorImpl::make("aten::tile", rt, {self})[0];
}

/// ['aten::tile', 'at::Tensor', 'tile_symint', '(const at::Tensor & self, c10::SymIntArrayRef dims)']
at::Tensor checkpoint_tile_symint(const at::Tensor & self, c10::SymIntArrayRef dims) {
  auto dims_ = dims.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::tile_symint(vec.at(0), dims_)};
    };
  return CheckpointTensorImpl::make("aten::tile", rt, {self})[0];
}

/// ['aten::resize', 'at::Tensor', 'resize', '(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_resize(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::resize(vec.at(0), size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::resize", rt, {self})[0];
}

/// ['aten::resize', 'at::Tensor', 'resize_symint', '(const at::Tensor & self, c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format=c10::nullopt)']
at::Tensor checkpoint_resize_symint(const at::Tensor & self, c10::SymIntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::resize_symint(vec.at(0), size_, memory_format)};
    };
  return CheckpointTensorImpl::make("aten::resize", rt, {self})[0];
}

/// ['aten::argmax', 'at::Tensor', 'argmax', '(const at::Tensor & self, c10::optional<int64_t> dim=c10::nullopt, bool keepdim=false)']
at::Tensor checkpoint_argmax(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::argmax(vec.at(0), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::argmax", rt, {self})[0];
}

/// ['aten::argmax.out', 'at::Tensor &', 'argmax_out', '(at::Tensor & out, const at::Tensor & self, c10::optional<int64_t> dim=c10::nullopt, bool keepdim=false)']
at::Tensor & checkpoint_argmax_out(at::Tensor & out, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::argmax_out(out, vec.at(1), dim, keepdim)};
    };
  return CheckpointTensorImpl::make("aten::argmax.out", rt, {out, self})[0];
}

/// ['aten::argmax.out', 'at::Tensor &', 'argmax_outf', '(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out)']
at::Tensor & checkpoint_argmax_outf(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::argmax_outf(vec.at(0), dim, keepdim, out)};
    };
  return CheckpointTensorImpl::make("aten::argmax.out", rt, {self, out})[0];
}

/// ['aten::bitwise_and.Tensor_out', 'at::Tensor &', 'bitwise_and_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_bitwise_and_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::bitwise_and_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::bitwise_and.Tensor_out', 'at::Tensor &', 'bitwise_and_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_bitwise_and_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::bitwise_and_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::bitwise_and.Scalar_out', 'at::Tensor &', 'bitwise_and_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_bitwise_and_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::bitwise_and_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar_out", rt, {out, self})[0];
}

/// ['aten::bitwise_and.Scalar_out', 'at::Tensor &', 'bitwise_and_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_bitwise_and_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::bitwise_and_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar_out", rt, {self, out})[0];
}

/// ['aten::bitwise_and.Scalar', 'at::Tensor', 'bitwise_and', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_bitwise_and(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::bitwise_and(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar", rt, {self})[0];
}

/// ['aten::bitwise_and.Scalar_Tensor', 'at::Tensor', 'bitwise_and', '(const at::Scalar & self, const at::Tensor & other)']
at::Tensor checkpoint_bitwise_and(const at::Scalar & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::bitwise_and(self, vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar_Tensor", rt, {other})[0];
}

/// ['aten::bitwise_and.Tensor', 'at::Tensor', 'bitwise_and', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_bitwise_and(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::bitwise_and(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Tensor", rt, {self, other})[0];
}

/// ['aten::bitwise_and.Scalar_Tensor_out', 'at::Tensor &', 'bitwise_and_out', '(at::Tensor & out, const at::Scalar & self, const at::Tensor & other)']
at::Tensor & checkpoint_bitwise_and_out(at::Tensor & out, const at::Scalar & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::bitwise_and_out(out, self, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar_Tensor_out", rt, {out, other})[0];
}

/// ['aten::bitwise_and.Scalar_Tensor_out', 'at::Tensor &', 'bitwise_and_outf', '(const at::Scalar & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_bitwise_and_outf(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::bitwise_and_outf(self, vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::bitwise_and.Scalar_Tensor_out", rt, {other, out})[0];
}

/// ['aten::_reshape_alias', 'at::Tensor', '_reshape_alias', '(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride)']
at::Tensor checkpoint__reshape_alias(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_reshape_alias(vec.at(0), size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::_reshape_alias", rt, {self})[0];
}

/// ['aten::_reshape_alias', 'at::Tensor', '_reshape_alias_symint', '(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride)']
at::Tensor checkpoint__reshape_alias_symint(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto size_ = size.vec();
  auto stride_ = stride.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::_reshape_alias_symint(vec.at(0), size_, stride_)};
    };
  return CheckpointTensorImpl::make("aten::_reshape_alias", rt, {self})[0];
}

/// ['aten::permute', 'at::Tensor', 'permute', '(const at::Tensor & self, at::IntArrayRef dims)']
at::Tensor checkpoint_permute(const at::Tensor & self, at::IntArrayRef dims) {
  auto dims_ = dims.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::permute(vec.at(0), dims_)};
    };
  return CheckpointTensorImpl::make("aten::permute", rt, {self})[0];
}

/// ['aten::narrow', 'at::Tensor', 'narrow', '(const at::Tensor & self, int64_t dim, int64_t start, int64_t length)']
// at::Tensor checkpoint_narrow(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::narrow(vec.at(0), dim, start, length)};
//     };
//   return CheckpointTensorImpl::make("aten::narrow", rt, {self})[0];
// }

/// ['aten::narrow', 'at::Tensor', 'narrow_symint', '(const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length)']
// at::Tensor checkpoint_narrow_symint(const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::narrow_symint(vec.at(0), dim, start, length)};
//     };
//   return CheckpointTensorImpl::make("aten::narrow", rt, {self})[0];
// }

/// ['aten::narrow.Tensor', 'at::Tensor', 'narrow', '(const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length)']
// at::Tensor checkpoint_narrow(const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::narrow(vec.at(0), dim, vec.at(1), length)};
//     };
//   return CheckpointTensorImpl::make("aten::narrow.Tensor", rt, {self, start})[0];
// }

/// ['aten::narrow.Tensor', 'at::Tensor', 'narrow_symint', '(const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length)']
// at::Tensor checkpoint_narrow_symint(const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length) {
//   rematerialize_function_t rt =
//     [=](const Tensors& vec) -> Tensors {
//       return {at::narrow_symint(vec.at(0), dim, vec.at(1), length)};
//     };
//   return CheckpointTensorImpl::make("aten::narrow.Tensor", rt, {self, start})[0];
// }

/// ['aten::reciprocal', 'at::Tensor', 'reciprocal', '(const at::Tensor & self)']
at::Tensor checkpoint_reciprocal(const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::reciprocal(vec.at(0))};
    };
  return CheckpointTensorImpl::make("aten::reciprocal", rt, {self})[0];
}

/// ['aten::reciprocal_', 'at::Tensor &', 'reciprocal_', '(at::Tensor & self)']
at::Tensor & checkpoint_reciprocal_(at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor self = vec.at(0);
      return {at::reciprocal_(self)};
    };
  return CheckpointTensorImpl::make("aten::reciprocal_", rt, {self})[0];
}

/// ['aten::reciprocal.out', 'at::Tensor &', 'reciprocal_out', '(at::Tensor & out, const at::Tensor & self)']
at::Tensor & checkpoint_reciprocal_out(at::Tensor & out, const at::Tensor & self) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::reciprocal_out(out, vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::reciprocal.out", rt, {out, self})[0];
}

/// ['aten::reciprocal.out', 'at::Tensor &', 'reciprocal_outf', '(const at::Tensor & self, at::Tensor & out)']
at::Tensor & checkpoint_reciprocal_outf(const at::Tensor & self, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::reciprocal_outf(vec.at(0), out)};
    };
  return CheckpointTensorImpl::make("aten::reciprocal.out", rt, {self, out})[0];
}

/// ['aten::repeat_interleave.Tensor', 'at::Tensor', 'repeat_interleave', '(const at::Tensor & repeats, c10::optional<int64_t> output_size=c10::nullopt)']
at::Tensor checkpoint_repeat_interleave(const at::Tensor & repeats, c10::optional<int64_t> output_size) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::repeat_interleave(vec.at(0), output_size)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.Tensor", rt, {repeats})[0];
}

/// ['aten::repeat_interleave.self_Tensor', 'at::Tensor', 'repeat_interleave', '(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim=c10::nullopt, c10::optional<int64_t> output_size=c10::nullopt)']
at::Tensor checkpoint_repeat_interleave(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::repeat_interleave(vec.at(0), vec.at(1), dim, output_size)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.self_Tensor", rt, {self, repeats})[0];
}

/// ['aten::repeat_interleave.self_int', 'at::Tensor', 'repeat_interleave', '(const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim=c10::nullopt, c10::optional<int64_t> output_size=c10::nullopt)']
at::Tensor checkpoint_repeat_interleave(const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::repeat_interleave(vec.at(0), repeats, dim, output_size)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.self_int", rt, {self})[0];
}

/// ['aten::repeat_interleave.self_int', 'at::Tensor', 'repeat_interleave_symint', '(const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim=c10::nullopt, c10::optional<int64_t> output_size=c10::nullopt)']
at::Tensor checkpoint_repeat_interleave_symint(const at::Tensor & self, c10::SymInt repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::repeat_interleave_symint(vec.at(0), repeats, dim, output_size)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.self_int", rt, {self})[0];
}

/// ['aten::repeat_interleave.Tensor_out', 'at::Tensor &', 'repeat_interleave_out', '(at::Tensor & out, const at::Tensor & repeats, c10::optional<int64_t> output_size=c10::nullopt)']
at::Tensor & checkpoint_repeat_interleave_out(at::Tensor & out, const at::Tensor & repeats, c10::optional<int64_t> output_size) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::repeat_interleave_out(out, vec.at(1), output_size)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.Tensor_out", rt, {out, repeats})[0];
}

/// ['aten::repeat_interleave.Tensor_out', 'at::Tensor &', 'repeat_interleave_outf', '(const at::Tensor & repeats, c10::optional<int64_t> output_size, at::Tensor & out)']
at::Tensor & checkpoint_repeat_interleave_outf(const at::Tensor & repeats, c10::optional<int64_t> output_size, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::repeat_interleave_outf(vec.at(0), output_size, out)};
    };
  return CheckpointTensorImpl::make("aten::repeat_interleave.Tensor_out", rt, {repeats, out})[0];
}

/// ['aten::gt.Scalar_out', 'at::Tensor &', 'gt_out', '(at::Tensor & out, const at::Tensor & self, const at::Scalar & other)']
at::Tensor & checkpoint_gt_out(at::Tensor & out, const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::gt_out(out, vec.at(1), other)};
    };
  return CheckpointTensorImpl::make("aten::gt.Scalar_out", rt, {out, self})[0];
}

/// ['aten::gt.Scalar_out', 'at::Tensor &', 'gt_outf', '(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)']
at::Tensor & checkpoint_gt_outf(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(1);
      return {at::gt_outf(vec.at(0), other, out)};
    };
  return CheckpointTensorImpl::make("aten::gt.Scalar_out", rt, {self, out})[0];
}

/// ['aten::gt.Scalar', 'at::Tensor', 'gt', '(const at::Tensor & self, const at::Scalar & other)']
at::Tensor checkpoint_gt(const at::Tensor & self, const at::Scalar & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::gt(vec.at(0), other)};
    };
  return CheckpointTensorImpl::make("aten::gt.Scalar", rt, {self})[0];
}

/// ['aten::gt.Tensor_out', 'at::Tensor &', 'gt_out', '(at::Tensor & out, const at::Tensor & self, const at::Tensor & other)']
at::Tensor & checkpoint_gt_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::gt_out(out, vec.at(1), vec.at(2))};
    };
  return CheckpointTensorImpl::make("aten::gt.Tensor_out", rt, {out, self, other})[0];
}

/// ['aten::gt.Tensor_out', 'at::Tensor &', 'gt_outf', '(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)']
at::Tensor & checkpoint_gt_outf(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(2);
      return {at::gt_outf(vec.at(0), vec.at(1), out)};
    };
  return CheckpointTensorImpl::make("aten::gt.Tensor_out", rt, {self, other, out})[0];
}

/// ['aten::gt.Tensor', 'at::Tensor', 'gt', '(const at::Tensor & self, const at::Tensor & other)']
at::Tensor checkpoint_gt(const at::Tensor & self, const at::Tensor & other) {
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::gt(vec.at(0), vec.at(1))};
    };
  return CheckpointTensorImpl::make("aten::gt.Tensor", rt, {self, other})[0];
}

/// ['aten::full.names', 'at::Tensor', 'full', '(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, at::TensorOptions options={})']
at::Tensor checkpoint_full(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full(size_, fill_value, names, options)};
    };
  return CheckpointTensorImpl::make("aten::full.names", rt, {})[0];
}

/// ['aten::full.names', 'at::Tensor', 'full', '(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_full(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full(size_, fill_value, names, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::full.names", rt, {})[0];
}

/// ['aten::full', 'at::Tensor', 'full', '(at::IntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={})']
at::Tensor checkpoint_full(at::IntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full(size_, fill_value, options)};
    };
  return CheckpointTensorImpl::make("aten::full", rt, {})[0];
}

/// ['aten::full', 'at::Tensor', 'full', '(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_full(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full(size_, fill_value, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::full", rt, {})[0];
}

/// ['aten::full', 'at::Tensor', 'full_symint', '(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options={})']
at::Tensor checkpoint_full_symint(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::TensorOptions options) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full_symint(size_, fill_value, options)};
    };
  return CheckpointTensorImpl::make("aten::full", rt, {})[0];
}

/// ['aten::full', 'at::Tensor', 'full_symint', '(c10::SymIntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)']
at::Tensor checkpoint_full_symint(c10::SymIntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      return {at::full_symint(size_, fill_value, dtype, layout, device, pin_memory)};
    };
  return CheckpointTensorImpl::make("aten::full", rt, {})[0];
}

/// ['aten::full.out', 'at::Tensor &', 'full_out', '(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value)']
at::Tensor & checkpoint_full_out(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_out(out, size_, fill_value)};
    };
  return CheckpointTensorImpl::make("aten::full.out", rt, {out})[0];
}

/// ['aten::full.out', 'at::Tensor &', 'full_outf', '(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out)']
at::Tensor & checkpoint_full_outf(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_outf(size_, fill_value, out)};
    };
  return CheckpointTensorImpl::make("aten::full.out", rt, {out})[0];
}

/// ['aten::full.out', 'at::Tensor &', 'full_symint_out', '(at::Tensor & out, c10::SymIntArrayRef size, const at::Scalar & fill_value)']
at::Tensor & checkpoint_full_symint_out(at::Tensor & out, c10::SymIntArrayRef size, const at::Scalar & fill_value) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_symint_out(out, size_, fill_value)};
    };
  return CheckpointTensorImpl::make("aten::full.out", rt, {out})[0];
}

/// ['aten::full.out', 'at::Tensor &', 'full_symint_outf', '(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out)']
at::Tensor & checkpoint_full_symint_outf(c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_symint_outf(size_, fill_value, out)};
    };
  return CheckpointTensorImpl::make("aten::full.out", rt, {out})[0];
}

/// ['aten::full.names_out', 'at::Tensor &', 'full_out', '(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names)']
at::Tensor & checkpoint_full_out(at::Tensor & out, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_out(out, size_, fill_value, names)};
    };
  return CheckpointTensorImpl::make("aten::full.names_out", rt, {out})[0];
}

/// ['aten::full.names_out', 'at::Tensor &', 'full_outf', '(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, at::Tensor & out)']
at::Tensor & checkpoint_full_outf(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, at::Tensor & out) {
  auto size_ = size.vec();
  rematerialize_function_t rt =
    [=](const Tensors& vec) -> Tensors {
      Tensor out = vec.at(0);
      return {at::full_outf(size_, fill_value, names, out)};
    };
  return CheckpointTensorImpl::make("aten::full.names_out", rt, {out})[0];
}




}}