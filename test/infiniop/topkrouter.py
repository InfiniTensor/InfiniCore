import ctypes
from ctypes import c_uint64
import torch
import torch.nn as nn
import torch.nn.functional as F

from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    torch_device_map,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # x_shape, x_stride, select_experts
    ((1, 256), None, 8),
    ((3, 256), None, 8),
]

# w (weight) types
# Note: 'None' means the same as input dtype
_X_DTYPES = [InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.F16]  #
# x types used for testing
_VALUE_DTYPES = [InfiniDtype.F32]

# Form the test cases by appending each element of _X_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (x_dtype,) for test_case in _TEST_CASES_ for x_dtype in _X_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def tensorInfo(data):
    print(
        "data:  ",
        data.is_contiguous(),
        data.device,
        data.dtype,
        data.shape,
        data.stride(),
        data.data_ptr(),
        hex(data.data_ptr()),
    )


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, correction_bias, config=None):
        super().__init__()
        self.config = config
        self.top_k = 8  # config.num_experts_per_tok
        self.n_routed_experts = 256  # config.n_routed_experts
        self.routed_scaling_factor = 2.5  # config.routed_scaling_factor
        self.n_group = 8  # config.n_group
        self.topk_group = 4  # config.topk_group
        self.norm_topk_prob = True  # config.norm_topk_prob

        # self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        # self.weight = torch.rand(256, 7168) * 2 - 1

        # self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        self.e_score_correction_bias = torch.zeros(256, device="cuda")
        self.e_score_correction_bias[:] = correction_bias[:]

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(
            0
        )  # Size([1, 256])
        group_scores = (
            scores_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )

        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=True)[
            1
        ]  # Size([1, 4])
        group_mask = torch.zeros_like(group_scores)  # Size([1, 8])
        group_mask.scatter_(1, group_idx, 1)  # Size([1, 8])

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )  # Size([1, 256])
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=True)[
            1
        ]  # Size([1, 8])

        return topk_indices

    def forward(self, router_logits):
        # hidden_states = hidden_states.view(-1, 7168)
        # router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = router_logits.sigmoid()  # (1,256)
        scores = scores.to(torch.float32)

        topk_indices = self.get_topk_indices(scores)  # (1,8)
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


def torch_topkrouter(router_logits, correction_bias):
    lable_indices, lable_values = DeepseekV3TopkRouter(correction_bias)(router_logits)
    lable_indices = lable_indices.to(torch.int32)
    return lable_values, lable_indices


def test(
    handle,
    device,
    x_shape,
    x_stride,
    topk,
    x_dtype=InfiniDtype.F32,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing topkrouter on {InfiniDeviceNames[device]} with x_shape:{x_shape}"
        f"x_stride:{x_stride} w_dtype:{InfiniDtypeNames[x_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    data = torch.arange(0, x_shape[0] * x_shape[1]).reshape(x_shape)

    N, width = x_shape
    x = TestTensor(
        x_shape, data.stride(), x_dtype, device, scale=5.0, bias=-5.0, mode="random"
    )
    correction_bias = TestTensor(
        [x_shape[1]], [1], InfiniDtype.F32, device, mode="random"
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTopkrouterDescriptor(
            handle, ctypes.byref(descriptor), x.descriptor, correction_bias.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, correction_bias]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTopkrouterWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    values = torch.zeros(
        (N, topk), dtype=torch.float32, device=torch_device_map[x.device]
    )
    indices = torch.zeros(
        (N, topk), dtype=torch.int32, device=torch_device_map[x.device]
    )

    def lib_topkrouter():
        check_error(
            LIBINFINIOP.infiniopTopkrouter(
                descriptor,
                workspace.data(),
                workspace_size.value,
                values.data_ptr(),
                indices.data_ptr(),
                x.data(),
                correction_bias.data(),
                2.5,
                topk,
                None,
            )
        )

    lib_topkrouter()
    lable_values, lable_indices = torch_topkrouter(
        x.actual_tensor(), correction_bias.actual_tensor()
    )

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(lable_values, values, atol=atol, rtol=rtol)
        debug(lable_indices, indices, atol=atol, rtol=rtol)

    assert torch.allclose(lable_values, values, atol=atol, rtol=rtol)
    assert torch.allclose(lable_indices, lable_indices, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_topkrouter(x.actual_tensor().clone(), tokp), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topkrouter(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTopkrouterDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _VALUE_DTYPES)

    print("\033[92mTest passed!\033[0m")
