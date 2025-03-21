import torch
import torch.nn as nn
import math
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
is_weight_transposed = False

_TEST_CASES = []

MODELS = {
    "7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    # "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    # "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    # "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
}

# Loop through models and layers to generate the new _TEST_CASES
for _, layers in MODELS.items():
    for layer in layers:
        for batch in [1, 16]:
            _TEST_CASES.append(((batch, layer[0], layer[1], is_weight_transposed)))

# Data types used for testing
_TENSOR_DTYPES = [torch.float16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# ==============================================================================
#  Definitions
# ==============================================================================
class QuantizeGPTQDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopQuantizeGPTQDescriptor_t = POINTER(QuantizeGPTQDescriptor)


def quantize(x, scale, zero, maxq):
    if scale.shape[1] == 1:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
    else:
        group_size = x.shape[1] // scale.shape[1]
        y = torch.zeros_like(x)
        for j in range(scale.shape[1]):
            q = torch.clamp(
                torch.round(
                    x[:, j * group_size : (j + 1) * group_size] / scale[:, j : j + 1]
                )
                + zero[:, j : j + 1],
                0,
                maxq,
            )
            y[:, j * group_size : (j + 1) * group_size] = scale[:, j : j + 1] * (
                q - zero[:, j : j + 1]
            )
        return y


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits=4,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            # self.scale = self.scale.unsqueeze(1)
            # self.zero = self.zero.unsqueeze(1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


class GPTQ:

    def __init__(self, weight):
        self.weight = weight
        self.dev = self.weight.device

        self.rows = self.weight.shape[0]
        self.columns = self.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1):
        W = self.weight.clone()

        W = W.float()

        # tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H.to("cpu")).to(
            H.device
        )  # 对于CUDA来说，这个地方直接在CUDA上做cholesky分解可能会失败
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        num_groups = self.columns // group_size
        if group_size == -1:
            scale = self.quantizer.scale.clone()
            zero = self.quantizer.zero.clone()
        else:
            scale = torch.zeros(self.rows, num_groups)
            zero = torch.zeros(self.rows, num_groups)
        for index in range(self.columns // blocksize):
            i1 = index * blocksize
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        self.quantizer.find_params(
                            W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                        )
                        ind = index * blocksize // group_size + i // group_size

                        scale[:, ind : ind + 1] = self.quantizer.scale
                        zero[:, ind : ind + 1] = self.quantizer.zero

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        # print('error', torch.sum(Losses).item())

        self.weight = Q.reshape(self.weight.shape).to(self.weight.dtype)
        self.scale = scale.to(self.weight.dtype)
        self.zero = zero.to(self.weight.dtype)


def get_scale_zero(b, a, c, group_size):
    weight = b.clone()
    inp = a.clone()
    out = c.clone()
    gptq = GPTQ(weight)
    gptq.quantizer = Quantizer()
    gptq.quantizer.configure(perchannel=True, sym=False, mse=False)
    gptq.add_batch(inp, out)
    gptq.fasterquant(group_size=group_size)

    return (
        gptq.weight.to(weight.device),
        gptq.scale.to(weight.device),
        gptq.zero.to(weight.device),
    )


def pack(weight, scale, zero):
    intweight = torch.round((weight + zero) / scale).to(torch.int32)
    qweight = torch.zeros(
        [weight.shape[0], weight.shape[1] // 8], dtype=torch.int32, device=weight.device
    )
    for i in range(intweight.shape[1]):
        qweight[:, i // 8] |= intweight[:, i] << (4 * (i % 8))
    return qweight


# PyTorch implementation for matrix multiplication
def quantize_gptq(a, b):  # 昇腾芯片的CPU不支持转置计算
    ans = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(b.dtype)
    return ans


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    lib,
    handle,
    torch_device,
    M,
    K,
    N,
    is_weight_transposed,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing QuantizeGPTQ on {torch_device}" f" M:{M}, K:{K}, N:{N}, dtype:{dtype}"
    )
    torch.manual_seed(12)
    # Initialize tensors
    a = 1e0 * torch.randn([K, M], dtype=dtype).to(torch_device)
    layer = nn.Linear(K, N)
    b = 1e-3 * layer.weight.data.to(dtype).to(torch_device)
    c = torch.zeros([N, M], dtype=dtype).to(torch_device)

    group_size = -1
    num_groups = 1
    if group_size == -1:
        num_groups = 1
    else:
        num_groups = K // group_size
    if is_weight_transposed:
        ans = quantize_gptq(a.t(), b.t())
    else:
        ans = quantize_gptq(b, a)
    packed_weights = torch.zeros([N, K // 8], dtype=torch.int32).to(torch_device)
    s = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)
    z = torch.zeros([N, num_groups], dtype=dtype).to(torch_device)
    if torch_device == "cuda":
        b_ref, s, z = get_scale_zero(b, a.t(), c, group_size)
        z = torch.zeros_like(s)
        packed_weights = pack(b_ref, s, z)
        # print(s)

    a_tensor, b_tensor, c_tensor, s_tensor, z_tensor, packed_weights_tensor = (
        to_tensor(a, lib),
        to_tensor(b, lib),
        to_tensor(c, lib),
        to_tensor(s, lib),
        to_tensor(z, lib),
        to_tensor(packed_weights, lib),
    )

    descriptor = infiniopQuantizeGPTQDescriptor_t()
    check_error(
        lib.infiniopCreateQuantizeGPTQDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            packed_weights_tensor.descriptor,
            s_tensor.descriptor,
            z_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [
        a_tensor,
        b_tensor,
        c_tensor,
        s_tensor,
        z_tensor,
        packed_weights_tensor,
    ]:
        tensor.destroyDesc(lib)

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetQuantizeGPTQWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, a.device)

    # Execute infiniop quantize_gptq operator
    check_error(
        lib.infiniopQuantizeGPTQ(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            packed_weights_tensor.data,
            s_tensor.data,
            z_tensor.data,
            a_tensor.data,
            b_tensor.data,
            None,
        )
    )

    def lib_quantize_gptq():
        check_error(
            lib.infiniopQuantizeLinearGPTQ(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                c_tensor.data,
                a_tensor.data,
                packed_weights_tensor.data,
                s_tensor.data,
                z_tensor.data,
                None,
            )
        )

    lib_quantize_gptq()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    # tmpa = ans.flatten()
    # tmpc = c.flatten()
    # for i in range(tmpa.shape[0]):
    #     if abs(tmpa[i] - tmpc[i]) > atol + rtol * abs(tmpa[i]):
    #         print(tmpa[i], tmpc[i], abs(tmpa[i] - tmpc[i]), rtol * abs(tmpa[i]))
    #         break

    if DEBUG:
        debug(c, ans, atol=atol, rtol=rtol)
    assert torch.allclose(c, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        if(is_weight_transposed):
            profile_operation("PyTorch", lambda: quantize_gptq(a.t(), b.t()), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        else:
            profile_operation("PyTorch", lambda: quantize_gptq(b, a), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_quantize_gptq(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyQuantizeGPTQDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateQuantizeGPTQDescriptor.restype = c_int32
    lib.infiniopCreateQuantizeGPTQDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopQuantizeGPTQDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetQuantizeGPTQWorkspaceSize.restype = c_int32
    lib.infiniopGetQuantizeGPTQWorkspaceSize.argtypes = [
        infiniopQuantizeGPTQDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopQuantizeGPTQ.restype = c_int32
    lib.infiniopQuantizeGPTQ.argtypes = [
        infiniopQuantizeGPTQDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopQuantizeLinearGPTQ.restype = c_int32
    lib.infiniopQuantizeLinearGPTQ.argtypes = [
        infiniopQuantizeGPTQDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyQuantizeGPTQDescriptor.restype = c_int32
    lib.infiniopDestroyQuantizeGPTQDescriptor.argtypes = [
        infiniopQuantizeGPTQDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
