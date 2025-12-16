import math
import torch
from torch.autograd import Function
import flash_attn_2_cuda as flash_attn_cuda

import torch.nn.functional as F
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb, apply_rotary_pos_emb_backward

from megatron.core.parallel_state import get_global_memory_buffer, get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim, _reduce_scatter_along_first_dim
from megatron.core.tensor_parallel.random import _set_cuda_rng_state, get_cuda_rng_tracker
from megatron.core import mpu
from apex.normalization.fused_layer_norm import fused_layer_norm_cuda
import importlib
import fused_weight_gradient_mlp_cuda
import logging
import time

from megatron.model.fused_bias_gelu import bias_gelu, bias_gelu_back

def gather_input(input_t, recompute_flag=True, name="mpu"):
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_t

    dim_size = list(input_t.size())
    dim_size[0] = dim_size[0] * world_size

    if recompute_flag:
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input_t.dtype, name)
    else:
        all_gather_buffer = torch.empty(dim_size, dtype=input_t.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        all_gather_buffer, input_t, group=get_tensor_model_parallel_group()
    )
    return all_gather_buffer

swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> T swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)

gelu_fwd_codestring = """
template <typename T> T gelu_fwd(T x) {
    return float(x) * 0.5f * (1.0f + ::tanh(0.79788456f * float(x) * (1 + 0.044715f * float(x) * float(x))));
}
"""
gelu_bwd_codestring = """
template <typename T> T gelu_bwd(T dx, T x) {
    float tanh_out = ::tanh(0.79788456f * float(x) * (1 + 0.044715f * float(x) * float(x)));
    float ff = float(x) * 0.5f * ((1.0f - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * float(x) * float(x))) + 0.5f * (1.0f + tanh_out);
    return ff * float(dx);
}
"""
gelu_fwd = torch.cuda.jiterator._create_jit_fn(gelu_fwd_codestring)
gelu_bwd = torch.cuda.jiterator._create_jit_fn(gelu_bwd_codestring)

def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return swiglu_fwd(x[0], x[1])

def swiglu_back(grad_output, x):
    x = torch.chunk(x, 2, dim=-1)
    grad_x0, grad_x1 = swiglu_bwd(x[0], x[1], grad_output)
    return torch.concat((grad_x0, grad_x1), dim=-1)

def layernorm_forward(hidden_state, normalized_shape, weight, bias, epsilon):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return fused_layer_norm_cuda.forward_affine(
        hidden_state, normalized_shape, weight, bias, epsilon)

def layernorm_backward(grad_output, mean, invvar, input_t,
                       normalized_shape, weight, bias, epsilon):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return fused_layer_norm_cuda.backward_affine(
        grad_output.contiguous(), mean, invvar, input_t,
        normalized_shape, weight, bias, epsilon, False)

def rmsnorm_forward(hidden_state, normalized_shape, weight, epsilon):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return fused_layer_norm_cuda.rms_forward_affine(
        hidden_state, normalized_shape, weight, epsilon)

def rmsnorm_backward(grad_input, invvar, hidden_state, normalized_shape, norm_weight, epsilon):
    global fused_layer_norm_cuda
    if fused_layer_norm_cuda is None:
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
    return fused_layer_norm_cuda.rms_backward_affine(
            grad_input, invvar, hidden_state, normalized_shape, norm_weight, epsilon, False)

def custom_flash_attention_forward(query, key, value,
                                   s, hidden_size_per_attention_head,
                                   cu_seqlens_q, cu_seqlens_k,
                                   attention_dropout):
    causal = True
    softmax_scale = 1.0 / math.sqrt(hidden_size_per_attention_head)
    return_softmax = False
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    window_size = (-1, -1)

    return flash_attn_cuda.varlen_fwd(
        query, key, value, None,
        cu_seqlens_q, cu_seqlens_k, None, None,
        s, s, attention_dropout,
        softmax_scale, False, causal,
        window_size[0], window_size[1],
        return_softmax, None)

def custom_flash_attention_backward(grad_context_layer,
                                    query, key, value,
                                    out_padded, softmax_lse,
                                    cu_seqlens_q, cu_seqlens_k, rng_state,
                                    s, attention_dropout, hidden_size_per_attention_head):
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    causal = True
    softmax_scale = 1.0 / math.sqrt(hidden_size_per_attention_head)

    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    grad_context_layer, query, key, value, old_context_layer = [maybe_contiguous(x) for x in (grad_context_layer, query, key, value, out_padded)]
    window_size = (-1, -1)
    deterministic = False

    grad_query, grad_key, grad_value, softmax_d, = flash_attn_cuda.varlen_bwd(
        grad_context_layer, query, key, value,
        old_context_layer, softmax_lse,
        grad_query, grad_key, grad_value,
        cu_seqlens_q, cu_seqlens_k, None,
        s, s, attention_dropout,
        softmax_scale, False, causal,
        window_size[0], window_size[1], deterministic,
        None, rng_state)
    grad_query = grad_query[..., :grad_context_layer.shape[-1]]  # We could have padded the head dimension
    grad_key = grad_key[..., :grad_context_layer.shape[-1]]  # We could have padded the head dimension
    grad_value = grad_value[..., :grad_context_layer.shape[-1]]  # We could have padded the head dimension
    return grad_query, grad_key, grad_value

def linear_mlp2_backward(grad_output, weight, total_input):
    grad_output = grad_output.contiguous()
    grad_input = grad_output.matmul(weight)

    grad_output = grad_output.view(
        grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
    )
    total_input = total_input.reshape(
        total_input.shape[0] * total_input.shape[1], total_input.shape[2]
    )

    if weight.main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, weight.main_grad
        )
    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            total_input, grad_output, weight.main_grad
        )
    return grad_input

def linear_mlp1_backward(grad_output,
                         weight,
                         total_input):
    grad_output = grad_output.contiguous()
    grad_input = grad_output.matmul(weight)

    grad_output = grad_output.view(
        grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
    )
    total_input = total_input.reshape(
        total_input.shape[0] * total_input.shape[1], total_input.shape[2]
    )

    dim_size = list(grad_input.size())
    dim_size[0] //= get_tensor_model_parallel_world_size()
    sub_grad_input = torch.empty(
        dim_size,
        dtype=total_input.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False
    )
    handle = torch.distributed._reduce_scatter_base(
        sub_grad_input,
        grad_input,
        group=get_tensor_model_parallel_group(),
        async_op=True
    )
    if weight.main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, weight.main_grad
        )
    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            total_input, grad_output, weight.main_grad
        )

    handle.wait()
    return sub_grad_input

def linear_weight_fused_gradient(grad_output, total_input, weight):
    if weight.main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
            total_input, grad_output, weight.main_grad
        )
    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
            total_input, grad_output, weight.main_grad
        )

class AttentionFunction(Function):
    @staticmethod
    def forward(ctx,
                hidden_state,
                norm_weight,
                norm_bias,
                query_weight,
                query_bias,
                key_weight,
                key_bias,
                value_weight,
                value_bias,
                atten_dense_weight,
                recompute_config,
                hidden_size_per_attention_head,
                args,
                rotary_pos_emb=None):

        saved_tensors = []
        ctx.input_shape = hidden_state.size()
        ctx.args = args
        ctx.recompute_config = recompute_config
        hidden_state = hidden_state.contiguous()
        saved_tensors.append(hidden_state)

        if args.position_embedding_type == "rope":
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)
            ctx.rotary_pos_emb = rotary_pos_emb

        # layernorm/rmsnorm
        assert not args.apply_layernorm_1p

        saved_tensors.append(norm_weight)
        saved_tensors.append(norm_bias)

        normalized_shape = torch.Size((args.hidden_size, ))
        if args.normalization == "LayerNorm":
            norm_output, mean, invvar = layernorm_forward(
                hidden_state, normalized_shape, norm_weight, norm_bias, args.layernorm_epsilon)
            if not recompute_config["attn_norm"]:
                saved_tensors.append(norm_output)
                saved_tensors.append(mean)
                saved_tensors.append(invvar)
        else: # RMSNORM
            norm_output, invvar = rmsnorm_forward(
                hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)
            if not recompute_config["attn_norm"]:
                saved_tensors.append(norm_output)
                saved_tensors.append(invvar)

        if args.debug_correctness:
            logging.info(f"attn norm output: {norm_output}")

        # gather
        total_input = gather_input(norm_output, recompute_config["attn_gather"])
        if not recompute_config["attn_gather"]:
            saved_tensors.append(total_input)
        if args.debug_correctness:
            logging.info(f"attn_gather: {total_input}")

        # query unit
        saved_tensors.append(query_weight)
        saved_tensors.append(query_bias)

        query = torch.matmul(total_input, query_weight.t())
        if args.add_bias_linear:
            query = query + query_bias
        if args.debug_correctness:
            torch.cuda.synchronize()
            logging.info(f"query before rope {query.shape}: {query}")
        # [sq, b, hn, hs]
        query = query.view(query.size(0), query.size(1), -1, hidden_size_per_attention_head)
        s = query.size(0)
        b = query.size(1)
        hn = query.size(2)
        hs = query.size(3)
        if args.position_embedding_type == "rope":
            q_pos_emb, _ = rotary_pos_emb
            query = apply_rotary_pos_emb(query, q_pos_emb)
            if args.debug_correctness:
                torch.cuda.synchronize()
                logging.info(f"query after rope {query.shape}: {query}")
        # [b * s, hn, hs]
        query = query.transpose(0, 1).reshape(-1, hn, hs)

        if not recompute_config['attn_query']:
            saved_tensors.append(query)
        if args.debug_correctness:
            torch.cuda.synchronize()
            logging.info(f"query {query.shape}: {query}")

        # key unit
        saved_tensors.append(key_weight)
        saved_tensors.append(key_bias)
        key = torch.matmul(total_input, key_weight.t())
        if args.add_bias_linear:
            key = key + key_bias
        if not recompute_config['attn_key']:
            saved_tensors.append(key)

        key = key.view(key.size(0), key.size(1), -1, hidden_size_per_attention_head)
        ctx.rep_num = hn // key.size(2)
        if ctx.rep_num > 1:
            key = key.repeat_interleave(hn // key.size(2), dim=2)
        if args.position_embedding_type == "rope":
            _, k_pos_emb = rotary_pos_emb
            key = apply_rotary_pos_emb(key, k_pos_emb)
        # [b * s, hn, hs]
        key = key.transpose(0, 1).reshape(-1, hn, hs)

        if args.debug_correctness:
            logging.info(f"key {key.shape}: {key}")

        # value unit
        saved_tensors.append(value_weight)
        saved_tensors.append(value_bias)
        value = torch.matmul(total_input, value_weight.t())
        if args.add_bias_linear:
            value = value + value_bias
        if not recompute_config['attn_value']:
            saved_tensors.append(value)

        value = value.view(value.size(0), value.size(1), -1, hidden_size_per_attention_head)
        if ctx.rep_num > 1:
            value = value.repeat_interleave(hn // value.size(2), dim=2)
        # [b * s, hn, hs]
        value = value.transpose(0, 1).reshape(-1, hn, hs)

        if args.debug_correctness:
            logging.info(f"value {value.shape}: {value}")

        # flash attention
        # Copy the rng states.
        fwd_cpu_rng_state = torch.get_rng_state()
        fwd_cuda_rng_state = torch.cuda.get_rng_state()
        fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        cu_seqlens_q = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=torch.cuda.current_device())
        cu_seqlens_k = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=torch.cuda.current_device())
        context_layer, query, key, value, out_padded, softmax_lse, S_dmask, rng_state = custom_flash_attention_forward(
            query, key, value,
            s, hidden_size_per_attention_head,
            cu_seqlens_q, cu_seqlens_k,
            args.attention_dropout)
        # change view [b, s, hn, hn]
        context_layer = context_layer.view(b, s, hn, hs)
        # [s, b, hidden]
        context_layer = context_layer.transpose(0, 1).reshape(s, b, -1)

        if not recompute_config['attn_fa']:
            saved_tensors.append(out_padded)
            saved_tensors.append(softmax_lse)
            saved_tensors.append(cu_seqlens_q)
            saved_tensors.append(cu_seqlens_k)
            saved_tensors.append(rng_state)
            saved_tensors.append(context_layer)
        else:
            ctx.fwd_cpu_rng_state = fwd_cpu_rng_state
            ctx.fwd_cuda_rng_state = fwd_cuda_rng_state
            ctx.fwd_cuda_rng_state_tracker = fwd_cuda_rng_state_tracker

        if args.debug_correctness:
            logging.info(f"context layer: {context_layer}")

        # attn dense unit
        saved_tensors.append(atten_dense_weight)
        result = torch.matmul(context_layer, atten_dense_weight.t())
        result = _reduce_scatter_along_first_dim(result)

        ctx.save_for_backward(*saved_tensors)
        ctx.hidden_size_per_attention_head = hidden_size_per_attention_head
        ctx.b = b
        ctx.s = s
        ctx.hn = hn
        ctx.hs = hs

        if args.debug_correctness:
            logging.info(f"result: {result}")

        return result

    @staticmethod
    def backward(ctx, grad_output):
        recompute_config = ctx.recompute_config
        hidden_size_per_attention_head = ctx.hidden_size_per_attention_head
        b = ctx.b
        s = ctx.s
        hn = ctx.hn
        hs = ctx.hs
        rep_num = ctx.rep_num
        args = ctx.args
        if args.position_embedding_type == "rope":
            rotary_pos_emb = ctx.rotary_pos_emb

        # compute forward
        current_idx = 0
        hidden_state = ctx.saved_tensors[current_idx]

        # rms norm
        current_idx += 1
        norm_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        norm_bias = ctx.saved_tensors[current_idx]

        normalized_shape = torch.Size((args.hidden_size,))
        if recompute_config["attn_norm"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            if args.normalization == "LayerNorm":
                norm_output, mean, invvar = layernorm_forward(
                    hidden_state, normalized_shape, norm_weight, norm_bias, args.layernorm_epsilon)
            else: # RMSNORM
                norm_output, invvar = rmsnorm_forward(
                    hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_norm: {end - begin}")
        else:
            if args.normalization == "LayerNorm":
                current_idx += 1
                norm_output = ctx.saved_tensors[current_idx]
                current_idx += 1
                mean = ctx.saved_tensors[current_idx]
                current_idx += 1
                invvar = ctx.saved_tensors[current_idx]
            else: # RMSNORM
                current_idx += 1
                norm_output = ctx.saved_tensors[current_idx]
                current_idx += 1
                invvar = ctx.saved_tensors[current_idx]

        if recompute_config["attn_gather"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            total_input = gather_input(norm_output, recompute_flag=True, name="mpu")

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_gather: {end - begin}")
        else:
            current_idx += 1
            total_input = ctx.saved_tensors[current_idx]

        # query unit
        current_idx += 1
        query_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        query_bias = ctx.saved_tensors[current_idx]
        if recompute_config["attn_query"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            query = torch.matmul(total_input, query_weight.t())
            if args.add_bias_linear:
                query = query + query_bias
            query = query.view(query.size(0), query.size(1), -1, hidden_size_per_attention_head)
            if args.position_embedding_type == "rope":
                q_pos_emb, _ = rotary_pos_emb
                query = apply_rotary_pos_emb(query, q_pos_emb)
            query = query.transpose(0, 1).reshape(-1, hn, hs)

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_query: {end - begin}")
        else:
            current_idx += 1
            query = ctx.saved_tensors[current_idx]

        # key unit
        current_idx += 1
        key_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        key_bias = ctx.saved_tensors[current_idx]
        if recompute_config["attn_key"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            key = torch.matmul(total_input, key_weight.t())
            if args.add_bias_linear:
                key = key + key_bias

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_key: {end - begin}")
        else:
            current_idx += 1
            key = ctx.saved_tensors[current_idx]

        key = key.view(key.size(0), key.size(1), -1, hidden_size_per_attention_head)
        if ctx.rep_num > 1:
            key = key.repeat_interleave(hn // key.size(2), dim=2)
        if args.position_embedding_type == "rope":
            _, k_pos_emb = rotary_pos_emb
            key = apply_rotary_pos_emb(key, k_pos_emb)
        key = key.transpose(0, 1).reshape(-1, hn, hs)

        # value unit
        current_idx += 1
        value_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        value_bias = ctx.saved_tensors[current_idx]
        if recompute_config["attn_value"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            value = torch.matmul(total_input, value_weight.t())
            if args.add_bias_linear:
                value = value + value_bias

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_value: {end - begin}")
        else:
            current_idx += 1
            value = ctx.saved_tensors[current_idx]

        value = value.view(value.size(0), value.size(1), -1, hidden_size_per_attention_head)
        if ctx.rep_num > 1:
            value = value.repeat_interleave(hn // value.size(2), dim=2)
        value = value.transpose(0, 1).reshape(-1, hn, hs)

        # bmm unit
        if recompute_config["attn_fa"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            # Store the current states.
            bwd_cpu_rng_state = torch.get_rng_state()
            bwd_cuda_rng_state = torch.cuda.get_rng_state()
            bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

            # Set the states to what it used to be before the forward pass.
            torch.set_rng_state(ctx.fwd_cpu_rng_state)
            _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

            cu_seqlens_q = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=torch.cuda.current_device())
            cu_seqlens_k = torch.arange(0, (b + 1) * s, step=s, dtype=torch.int32, device=torch.cuda.current_device())
            context_layer, query, key, value, out_padded, softmax_lse, S_dmask, rng_state = custom_flash_attention_forward(
                query, key, value,
                s, hidden_size_per_attention_head,
                cu_seqlens_q, cu_seqlens_k,
                args.attention_dropout)

            # Set the states back to what it was at the start of this function.
            torch.set_rng_state(bwd_cpu_rng_state)
            _set_cuda_rng_state(bwd_cuda_rng_state)
            get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

            context_layer = context_layer.view(b, s, hn, hs)
            context_layer = context_layer.transpose(0, 1).reshape(s, b, -1)

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"attn_fa: {end - begin}")
        else:
            current_idx += 1
            out_padded = ctx.saved_tensors[current_idx]
            current_idx += 1
            softmax_lse = ctx.saved_tensors[current_idx]
            current_idx += 1
            cu_seqlens_q = ctx.saved_tensors[current_idx]
            current_idx += 1
            cu_seqlens_k = ctx.saved_tensors[current_idx]
            current_idx += 1
            rng_state = ctx.saved_tensors[current_idx]
            current_idx += 1
            context_layer = ctx.saved_tensors[current_idx]


        # Start Backward Process
        # attn dense unit
        current_idx += 1
        atten_dense_weight = ctx.saved_tensors[current_idx]

        # compute backward
        grad_output = _gather_along_first_dim(grad_output)
        grad_output.contiguous()

        # [sq, b, h]
        grad_context_layer = linear_mlp2_backward(grad_output, atten_dense_weight, context_layer)

        # compute backward of context layer
        # [sq, b, hp] --> [sq, b, np, hn]
        grad_context_layer = grad_context_layer.transpose(0, 1).view(b, s, hn, hs)
        grad_context_layer = grad_context_layer.reshape(-1, hn, hs)

        grad_query, grad_key, grad_value = custom_flash_attention_backward(
            grad_context_layer,
            query, key, value,
            out_padded, softmax_lse,
            cu_seqlens_q, cu_seqlens_k, rng_state,
            s, args.attention_dropout, hidden_size_per_attention_head)

        # compute backward of value
        # [b * s,hn, hs] --> [s, b * hn, hs]
        grad_value = grad_value.view(b, s, hn, hs).transpose(0, 1)
        if ctx.rep_num > 1:
            grad_value = grad_value.reshape(b, s, hn // rep_num, rep_num, hs).sum(dim=-2)
        grad_value = grad_value.reshape(b * s, -1).contiguous()
        if args.add_bias_linear:
            grad_value_bias = grad_value.sum(dim=0)
        else:
            grad_value_bias = None

        grad_input = grad_value.matmul(value_weight)
        total_input = total_input.view(total_input.size(0) * total_input.size(1), -1).contiguous()
        linear_weight_fused_gradient(grad_value, total_input, value_weight)

        # compute backward of key
        grad_key = grad_key.view(b, s, hn, hs).transpose(0, 1)
        # rotary backward
        if args.position_embedding_type == "rope":
            _, k_pos_emb = rotary_pos_emb
            grad_key = apply_rotary_pos_emb_backward(grad_key, k_pos_emb)
        if ctx.rep_num > 1:
            grad_key = grad_key.reshape(b, s, hn // rep_num, rep_num, hs).sum(dim=-2)
        grad_key = grad_key.reshape(b * s, -1).contiguous()
        if args.add_bias_linear:
            grad_key_bias = grad_key.sum(dim=0)
        else:
            grad_key_bias = None

        grad_input = torch.addmm(
            grad_input,
            grad_key,
            key_weight)
        linear_weight_fused_gradient(grad_key, total_input, key_weight)

        # compute_backward of query
        grad_query = grad_query.view(b, s, hn, hs).transpose(0, 1)
        if args.position_embedding_type == "rope":
            q_pos_emb, _ = rotary_pos_emb
            grad_query = apply_rotary_pos_emb_backward(grad_query, q_pos_emb)
        grad_query = grad_query.reshape(b * s, -1).contiguous()
        if args.add_bias_linear:
            grad_query_bias = grad_query.sum(dim=0)
        else:
            grad_query_bias = None

        grad_input = torch.addmm(
            grad_input,
            grad_query,
            query_weight)

        dim_size = list(grad_input.size())
        dim_size[0] //= get_tensor_model_parallel_world_size()
        sub_grad_input = torch.empty(
            dim_size,
            dtype=total_input.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False
        )
        handle = torch.distributed._reduce_scatter_base(
            sub_grad_input,
            grad_input,
            group=get_tensor_model_parallel_group(),
            async_op=True
        )
        if query_weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input, grad_query, query_weight.main_grad
            )
        elif query_weight.main_grad.dtype in (torch.float16, torch.bfloat16):
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input, grad_query, query_weight.main_grad
            )
        handle.wait()

        sub_grad_input = sub_grad_input.reshape(ctx.input_shape).contiguous()
        if args.normalization == "LayerNorm":
            grad_hidden_state, grad_norm_weight, grad_norm_bias = layernorm_backward(
                sub_grad_input, mean, invvar, hidden_state, normalized_shape,
                norm_weight, norm_bias, args.layernorm_epsilon)
        else:
            grad_hidden_state, grad_norm_weight = rmsnorm_backward(
                sub_grad_input, invvar, hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)
            grad_norm_bias = None

        return grad_hidden_state, grad_norm_weight, grad_norm_bias, \
            None, grad_query_bias, None, grad_key_bias, None, grad_value_bias, None, \
            None, None, None, None

class LinearFunction(Function):
    @staticmethod
    def forward(ctx,
                hidden_state,
                norm_weight,
                norm_bias,
                h_to_4h_weight,
                h_to_4h_bias,
                back_to_h_weight,
                recompute_config,
                args):

        saved_tensors = []
        ctx.args = args
        ctx.recompute_config = recompute_config
        hidden_state = hidden_state.contiguous()
        saved_tensors.append(hidden_state)

        assert args.gradient_accumulation_fusion

        # layernorm
        assert not args.apply_layernorm_1p

        saved_tensors.append(norm_weight)
        saved_tensors.append(norm_bias)

        normalized_shape = torch.Size((args.hidden_size, ))
        if args.normalization == "LayerNorm":
            norm_output, mean, invvar = layernorm_forward(
                hidden_state, normalized_shape, norm_weight, norm_bias, args.layernorm_epsilon)
            if not recompute_config["linear_norm"]:
                saved_tensors.append(norm_output)
                saved_tensors.append(mean)
                saved_tensors.append(invvar)
        else: # RMSNORM
            norm_output, invvar = rmsnorm_forward(
                hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)
            if not recompute_config["linear_norm"]:
                saved_tensors.append(norm_output)
                saved_tensors.append(invvar)

        if args.debug_correctness:
            logging.info(f"linear norm output: {norm_output}")

        # gather
        total_input = gather_input(norm_output, recompute_config["linear_gather"])
        if not recompute_config["linear_gather"]:
            saved_tensors.append(total_input)
        if args.debug_correctness:
            logging.info(f"linear_gather: {total_input}")

        # h_to_4h
        saved_tensors.append(h_to_4h_weight)
        saved_tensors.append(h_to_4h_bias)

        intermediate_result1 = torch.matmul(total_input, h_to_4h_weight.t())
        # input tensor is saved
        if not recompute_config['linear_mlp1']:
            saved_tensors.append(intermediate_result1)
        if args.debug_correctness:
            logging.info(f"intermediate_result1: {intermediate_result1}")

        # gelu unit
        if args.swiglu:
            if args.add_bias_linear:
                intermediate_result1 += h_to_4h_bias
            intermediate_result2 = swiglu(intermediate_result1)
        else:
            intermediate_result2 = bias_gelu(h_to_4h_bias, intermediate_result1)
        if not recompute_config['linear_activation']:
            saved_tensors.append(intermediate_result2)
        if args.debug_correctness:
            logging.info(f"activation: {intermediate_result2}")

        # linear2 unit
        saved_tensors.append(back_to_h_weight)

        intermediate_result3 = torch.matmul(intermediate_result2, back_to_h_weight.t())
        intermediate_result3 = _reduce_scatter_along_first_dim(intermediate_result3)

        # last tensor is sure to be saved

        ctx.save_for_backward(*saved_tensors)
        return intermediate_result3

    @staticmethod
    def backward(ctx, grad_output):
        recompute_config = ctx.recompute_config
        args = ctx.args

        #compute forward process of linear mlp1
        current_idx = 0
        hidden_state = ctx.saved_tensors[current_idx]

        origin_grad_output = grad_output

        # rms norm
        current_idx += 1
        norm_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        norm_bias = ctx.saved_tensors[current_idx]
        normalized_shape = torch.Size((args.hidden_size,))
        if recompute_config["linear_norm"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            if args.normalization == "LayerNorm":
                norm_output, mean, invvar = layernorm_forward(
                    hidden_state, normalized_shape, norm_weight, norm_bias, args.layernorm_epsilon)
            else: # RMSNORM
                norm_output, invvar = rmsnorm_forward(
                    hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"linear_norm: {end - begin}")
        else:
            if args.normalization == "LayerNorm":
                current_idx += 1
                norm_output = ctx.saved_tensors[current_idx]
                current_idx += 1
                mean = ctx.saved_tensors[current_idx]
                current_idx += 1
                invvar = ctx.saved_tensors[current_idx]
            else: # RMSNORM
                current_idx += 1
                norm_output = ctx.saved_tensors[current_idx]
                current_idx += 1
                invvar = ctx.saved_tensors[current_idx]

        if recompute_config["linear_gather"]:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()

            total_input = gather_input(norm_output, recompute_flag=True, name="mpu")

            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"linear_gather: {end - begin}")
        else:
            current_idx += 1
            total_input = ctx.saved_tensors[current_idx]

        current_idx += 1
        h_to_4h_weight = ctx.saved_tensors[current_idx]
        current_idx += 1
        h_to_4h_bias = ctx.saved_tensors[current_idx]
        if recompute_config['linear_mlp1']:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()
            intermediate_result1 = torch.matmul(total_input, h_to_4h_weight.t())
            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"linear_mlp1: {end - begin}")
        else:
            current_idx += 1
            intermediate_result1 = ctx.saved_tensors[current_idx]

        # compute forward process of swiglu
        if recompute_config['linear_activation']:
            if args.profile_recompute:
                torch.cuda.synchronize()
                begin = time.time()
            if args.swiglu:
                if args.add_bias_linear:
                    intermediate_result1 += h_to_4h_bias
                intermediate_result2 = swiglu(intermediate_result1)
            else:
                intermediate_result2 = bias_gelu(h_to_4h_bias, intermediate_result1)
            if args.profile_recompute:
                torch.cuda.synchronize()
                end = time.time()
                logging.info(f"linear_activation: {end - begin}")
        else:
            current_idx += 1
            intermediate_result2 = ctx.saved_tensors[current_idx]

        # compute forward process of linear mlp2
        current_idx += 1
        back_to_h_weight = ctx.saved_tensors[current_idx]

        # compute backward process of linear mlp2
        grad_output = _gather_along_first_dim(grad_output)
        grad_output = grad_output.contiguous()

        grad_result2 = linear_mlp2_backward(grad_output, back_to_h_weight, intermediate_result2)
        # grad back_to_h_weight is None

        # compute backward process of linear gelu
        if args.swiglu:
            grad_result1 = swiglu_back(grad_result2, intermediate_result1)
        else:
            grad_result1 = bias_gelu_back(grad_result2, h_to_4h_bias, intermediate_result1)
        if args.add_bias_linear:
            grad_h_to_4h_bias = grad_result1.sum(dim=0)
        else:
            grad_h_to_4h_bias = None

        # compute backward
        grad_input = linear_mlp1_backward(grad_result1,
                                          h_to_4h_weight,
                                          total_input)

        grad_input = grad_input.contiguous()
        if args.normalization == "LayerNorm":
            grad_hidden_state, grad_norm_weight, grad_norm_bias = layernorm_backward(
                grad_input, mean, invvar, hidden_state, normalized_shape,
                norm_weight, norm_bias, args.layernorm_epsilon)
        else:
            grad_hidden_state, grad_norm_weight = rmsnorm_backward(
                grad_input, invvar, hidden_state, normalized_shape, norm_weight, args.layernorm_epsilon)
            grad_norm_bias = None
        return grad_hidden_state, grad_norm_weight, grad_norm_bias, \
            None, grad_h_to_4h_bias, None, None, None
