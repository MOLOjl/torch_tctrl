import torch
import time
import os

DTR_ENABLE = True if int(os.environ.get('DTR_ENABLE', 0)) == 1 else False
RECORD_MEM_SNAPSHOT = True if int(os.environ.get('RECORD_MEM_SNAPSHOT', 0)) == 1 else False

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None

def NebulaFor3rdop(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        processed_inputs = []
        if DTR_ENABLE:
            for t in args:
                if isinstance(t, torch.Tensor):
                    t = t.decheckpoint()
                processed_inputs.append(t)
        else:
            processed_inputs = args
        result = func(*processed_inputs, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution cost time: {end_time - start_time}s")
        return result
    return wrapper

def MemoryProfiler(func):
    def wrapper(*args, **kwargs):
        if RECORD_MEM_SNAPSHOT:
            torch.cuda.memory._record_memory_history()
        result = func(*args, **kwargs)
        if RECORD_MEM_SNAPSHOT:
            torch.cuda.memory._dump_snapshot('flashut_nc.pickle')
        return result
    return wrapper

class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    @NebulaFor3rdop
    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output
    

@MemoryProfiler
def main():
    core_attention_flash = FlashSelfAttention(
        causal=True, attention_dropout=0
    )
    # q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    def init_tensor(B, S, H, D):
        t = torch.randn(B, S, H, D).to(device='cuda', dtype=torch.bfloat16)
        if DTR_ENABLE:
            t = t.checkpoint()
        return t
    B,S,H,D = 4,4096,32,128
    inputs = []
    for _ in range(3):
        inputs.append(init_tensor(B,S,H,D))
    out = core_attention_flash(*inputs)
    print(out.shape)

if __name__ == '__main__':
    if DTR_ENABLE:
        torch.init_dtb_manager()
    main()