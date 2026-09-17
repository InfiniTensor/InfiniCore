"""Exercise the reduction and reciprocal norm used by tensor-parallel models."""

import pytest
import torch

import infinicore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA device is required.")
def test_sum_descriptor_cache_distinguishes_axes():
    source = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    tensor = infinicore.from_torch(source)
    for axis in (0, 1, 0):
        output = torch.empty(4, device="cuda")
        torch.cuda.synchronize()
        infinicore.sum(tensor, dim=axis, out=infinicore.from_torch(output))
        infinicore.sync_device()
        torch.testing.assert_close(output, source.sum(axis))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA device is required.")
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
def test_sum_scalar_power_replay(keepdim, batch):
    device = infinicore.device("cuda", 0)
    infinicore.set_device(device)
    source = torch.linspace(0.25, 2.0, batch * 768, device="cuda").reshape(batch, 768)
    output = torch.empty((batch, 1) if keepdim else (batch,), device="cuda")
    input_tensor = infinicore.from_torch(source)
    output_tensor = infinicore.from_torch(output)
    torch.cuda.synchronize()

    infinicore.start_graph_recording(device)
    squared = infinicore.mul(input_tensor, input_tensor)
    reduced = infinicore.sum(squared, dim=1, keepdim=keepdim)
    infinicore.float_power(reduced, -0.5, out=output_tensor)
    graph = infinicore.stop_graph_recording()

    for scale in (1.0, 0.5, 3.0):
        source.mul_(scale)
        torch.cuda.synchronize()
        graph.run()
        infinicore.sync_stream()
        expected = source.square().sum(dim=1, keepdim=keepdim).rsqrt()
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
