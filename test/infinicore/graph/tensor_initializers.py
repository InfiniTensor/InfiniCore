import argparse

import infinicore
import torch


def assert_filled(tensor, expected, torch_device):
    actual = torch.empty(tuple(tensor.shape), dtype=torch.float32, device=torch_device)
    infinicore.from_torch(actual).copy_(tensor)
    infinicore.sync_stream()
    torch.testing.assert_close(actual, torch.full_like(actual, expected))


def test(device_name):
    torch_device = torch.device(device_name)
    device = infinicore.device(device_name, 0)
    shape = [8, 16]

    increment = infinicore.ones(shape, dtype=infinicore.float32, device=device)

    infinicore.start_graph_recording(device)
    zero_value = infinicore.zeros(shape, dtype=infinicore.float32, device=device)
    one_value = infinicore.ones(shape, dtype=infinicore.float32, device=device)
    infinicore.add(zero_value, increment, out=zero_value)
    infinicore.add(one_value, increment, out=one_value)
    graph = infinicore.stop_graph_recording()

    for _ in range(3):
        assert_filled(zero_value, 1.0, torch_device)
        assert_filled(one_value, 2.0, torch_device)
        graph.run()

    assert_filled(zero_value, 1.0, torch_device)
    assert_filled(one_value, 2.0, torch_device)
    print(f"tensor initializer graph replay on {device_name} ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cpu", action="store_true")
    group.add_argument("--iluvatar", action="store_true")
    args = parser.parse_args()
    test("cpu" if args.cpu else "cuda")
