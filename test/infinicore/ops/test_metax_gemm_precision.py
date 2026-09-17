"""Check MetaX GEMM precision in processes with independent descriptor caches."""

import os
import subprocess
import sys

import pytest
import torch
from infinicore.lib import _infinicore

import infinicore


@pytest.mark.skipif(
    _infinicore.get_device_count(_infinicore.Device.Type.METAX) == 0,
    reason="A MetaX device is required.",
)
@pytest.mark.parametrize("allow_tf32", [None, "0"])
def test_metax_gemm_precision(allow_tf32):
    env = os.environ.copy()
    if allow_tf32 is None:
        env.pop("INFINIOP_METAX_ALLOW_TF32", None)
    else:
        env["INFINIOP_METAX_ALLOW_TF32"] = allow_tf32
    result = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(20260917)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for rows in (1, 3, 257):
            x = torch.randn(rows, 768, generator=generator).to(dtype)
            weight = torch.randn(96, 768, generator=generator).to(dtype)
            expected = (x.double() @ weight.double().T).to(dtype)
            x_device, weight_device = x.cuda(), weight.cuda().T
            output = torch.empty((rows, 96), device="cuda", dtype=dtype)
            torch.cuda.synchronize()
            infinicore.matmul(
                infinicore.from_torch(x_device),
                infinicore.strided_from_blob(
                    weight_device.data_ptr(),
                    list(weight_device.shape),
                    list(weight_device.stride()),
                    dtype=infinicore.utils.to_infinicore_dtype(dtype),
                    device=infinicore.device("cuda", 0),
                ),
                out=infinicore.from_torch(output),
            )
            infinicore.sync_device()
            strict = (
                dtype == torch.float32 and os.getenv("INFINIOP_METAX_ALLOW_TF32") == "0"
            )
            tolerance = (2e-4, 1e-5) if strict else (0.1, 1e-2)
            torch.testing.assert_close(
                output.cpu(), expected, atol=tolerance[0], rtol=tolerance[1]
            )
