"""Last-axis concatenation must replay its producers and strided copies."""

import pytest
import torch

import infinicore


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A GPU is required.")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 4])
def test_last_axis_cat_replays_changed_inputs(dtype, batch):
    device = infinicore.device("cuda", 0)
    infinicore.set_device(device)
    source = (
        torch.linspace(-1, 1, batch * 129, device="cuda")
        .reshape(1, batch, 129)
        .to(dtype)
    )
    # Match projection views with a larger row stride than the sliced width.
    inputs = [source[..., :64], source[..., 64:96], source[..., 96:]]
    tensors = [
        infinicore.strided_from_blob(
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            dtype=infinicore.utils.to_infinicore_dtype(dtype),
            device=device,
        )
        for tensor in inputs
    ]
    output = torch.empty_like(source)
    target = infinicore.from_torch(output)
    torch.cuda.synchronize()

    infinicore.start_graph_recording(device)
    projected = infinicore.mul(tensors[0], tensors[0])
    joined = infinicore.cat([projected, *tensors[1:]], dim=-1)
    infinicore.mul(joined, joined, out=target)
    graph = infinicore.stop_graph_recording()

    for scale in (1.0, -0.5, 2.0):
        source.mul_(scale)
        torch.cuda.synchronize()
        graph.run()
        infinicore.sync_stream()
        expected = torch.cat([inputs[0].square(), *inputs[1:]], dim=-1).square()
        torch.testing.assert_close(output, expected)
