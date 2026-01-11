import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
# Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES = [
    # num_seqs, num_heads, num_kv_heads, head_size, block_size, max_step_len, num_rounds
    (1, 1, 1, 128, 8, 16, 1),
    (1, 4, 4, 128, 8, 16, 4),
    (2, 8, 8, 128, 16, 32, 2),
    (4, 16, 16, 128, 8, 64, 3),
    (8, 64, 64, 128, 8, 16, 5),
    (16, 128, 128, 128, 8, 16, 4),
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 2e-2, "rtol": 2e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 5
NUM_ITERATIONS = 10


# ==============================================================================
# Helper Classes & Reference Implementation
# ==============================================================================
class SimpleCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []

        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])

        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))

        self.request_to_len[request_id] = new_total_len
        return self.request_to_blocks[request_id], new_total_len


def ref_paged_attention_multi_turn(
    query_new, k_cache, v_cache, block_tables, seq_lens, cum_seq_lens_q, scale
):
    block_size = k_cache.shape[2]
    outputs = torch.zeros_like(query_new)
    num_seqs = len(cum_seq_lens_q) - 1
    for i in range(num_seqs):
        num_new = cum_seq_lens_q[i + 1].item() - cum_seq_lens_q[i].item()
        total_len = seq_lens[i].item()
        cache_len = seq_lens[i].item() - num_new

        table = block_tables[i]
        keys_all, values_all = [], []
        for j in range(total_len):
            b_id = table[j // block_size].item()
            off = j % block_size
            keys_all.append(k_cache[b_id, :, off, :])
            values_all.append(v_cache[b_id, :, off, :])

        K = torch.stack(keys_all, dim=0)
        V = torch.stack(values_all, dim=0)
        Q = query_new[cum_seq_lens_q[i] : cum_seq_lens_q[i + 1], :, :]

        scores = torch.einsum("qhd,khd->hqk", Q, K).float() * scale

        mask = torch.full((num_new, total_len), float("-inf"), device=Q.device)
        for q_idx in range(num_new):
            mask[q_idx, : cache_len + q_idx + 1] = 0.0

        scores = scores + mask.unsqueeze(0)
        attn_weights = torch.softmax(scores, dim=-1).to(Q.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, V)

        outputs[cum_seq_lens_q[i] : cum_seq_lens_q[i + 1], :, :] = out

    return outputs


# ==============================================================================
# Test Operator Implementation
# ==============================================================================
def test(
    handle,
    device,
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_step_len,
    num_rounds,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing PagedAttentionPrefill on {InfiniDeviceNames[device]} with "
        f"seqs:{num_seqs}, heads:{num_heads}, head_size:{head_size}, "
        f"block:{block_size}, max_step_len:{max_step_len}, num_rounds:{num_rounds}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # 1. Initialize persistent resources
    num_blocks = 8192
    manager = SimpleCacheManager(num_blocks, block_size)
    scale = head_size**-0.5

    k_cache = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )
    v_cache = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )

    # Multi-turn testing loop
    for r in range(num_rounds):
        # Prepare dynamic inputs for this round
        query_lens_cpu = torch.randint(
            1, max_step_len + 1, (num_seqs,), dtype=torch.int64
        )

        q_total_tokens = query_lens_cpu.sum().item()
        q_packed_tensors = torch.zeros(q_total_tokens, num_heads, head_size)

        seq_lens_list = []
        all_block_tables = []

        cum_seq_lens_q_list = []
        cum_q_lens = 0
        for i in range(num_seqs):
            cum_seq_lens_q_list.append(cum_q_lens)

            cur_q_len = query_lens_cpu[i].item()
            table, total_len = manager.allocate_slots(i, cur_q_len)
            cur_seq_lens = total_len - cur_q_len
            seq_lens_list.append(total_len)
            all_block_tables.append(table)

            # Simulated KV insertion
            k_new = torch.randn(cur_q_len, num_kv_heads, head_size)
            v_new = torch.randn(cur_q_len, num_kv_heads, head_size)
            q_val = torch.randn(cur_q_len, num_heads, head_size)
            q_packed_tensors[cum_q_lens : cum_q_lens + cur_q_len] = q_val

            cum_q_lens = cum_q_lens + cur_q_len

            for t in range(cur_q_len):
                logical_pos = cur_seq_lens + t
                b_id = table[logical_pos // block_size]
                off = logical_pos % block_size
                k_cache.torch_tensor()[b_id, :, off, :] = k_new[t]
                v_cache.torch_tensor()[b_id, :, off, :] = v_new[t]

        cum_seq_lens_q_list.append(cum_q_lens)

        k_cache.actual_tensor().copy_(k_cache._torch_tensor)
        v_cache.actual_tensor().copy_(v_cache._torch_tensor)

        # 2. Wrap tensors for Infiniop
        q_new = TestTensor.from_torch(q_packed_tensors, dtype, device)
        out = TestTensor.from_torch(q_packed_tensors, dtype, device)
        out.actual_tensor().zero_()

        seq_lens = TestTensor.from_torch(
            torch.tensor(seq_lens_list, dtype=torch.int64), InfiniDtype.I64, device
        )

        cum_seq_lens_q = TestTensor.from_torch(
            torch.tensor(cum_seq_lens_q_list, dtype=torch.int64),
            InfiniDtype.I64,
            device,
        )

        max_blocks = max(len(t) for t in all_block_tables)
        padded_tables = [t + [0] * (max_blocks - len(t)) for t in all_block_tables]
        block_tables = TestTensor.from_torch(
            torch.tensor(padded_tables, dtype=torch.int64), InfiniDtype.I64, device
        )

        # 3. Reference Calculation
        def torch_paged_attention_multi_turn():
            return ref_paged_attention_multi_turn(
                q_new.torch_tensor(),
                k_cache.torch_tensor(),
                v_cache.torch_tensor(),
                block_tables.torch_tensor(),
                seq_lens.torch_tensor(),
                cum_seq_lens_q.torch_tensor(),
                scale,
            )

        ans = torch_paged_attention_multi_turn()

        # 4. Infiniop Operator Execution
        descriptor = infiniopOperatorDescriptor_t()
        check_error(
            LIBINFINIOP.infiniopCreatePagedAttentionPrefillDescriptor(
                handle,
                ctypes.byref(descriptor),
                out.descriptor,
                q_new.descriptor,
                k_cache.descriptor,
                v_cache.descriptor,
                block_tables.descriptor,
                seq_lens.descriptor,
                cum_seq_lens_q.descriptor,
                None,
                scale,
            )
        )

        workspace_size = c_uint64(0)
        check_error(
            LIBINFINIOP.infiniopGetPagedAttentionPrefillWorkspaceSize(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, device)

        def lib_attn():
            check_error(
                LIBINFINIOP.infiniopPagedAttentionPrefill(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    out.data(),
                    q_new.data(),
                    k_cache.data(),
                    v_cache.data(),
                    block_tables.data(),
                    seq_lens.data(),
                    cum_seq_lens_q.data(),
                    None,
                    None,
                )
            )

        lib_attn()
        if sync:
            sync()

        # 5. Validation
        atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
        if DEBUG:
            debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)

        assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

        # Profiling
        if PROFILE:
            profile_operation(
                f"Torch_R{r}",
                lambda: torch_paged_attention_multi_turn(),
                device,
                NUM_PRERUN,
                NUM_ITERATIONS,
            )
            profile_operation(
                f"  Lib_R{r}", lambda: lib_attn(), device, NUM_PRERUN, NUM_ITERATIONS
            )

        check_error(
            LIBINFINIOP.infiniopDestroyPagedAttentionPrefillDescriptor(descriptor)
        )


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
