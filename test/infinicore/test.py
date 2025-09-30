import infinicore
import torch


def test():
    shape = [2, 3, 4]
    shape2 = [3, 4, 2]
    torch_tensor_ans = torch.rand(shape, dtype=torch.float32, device="cpu")
    torch_tensor_result = torch.zeros(shape, dtype=torch.float32, device="cpu")

    t_cpu = infinicore.from_blob(
        torch_tensor_ans.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )

    t_gpu = t_cpu.to(infinicore.device("cuda", 0))

    t_gpu = t_gpu.permute([1, 2, 0])

    t_gpu2 = infinicore.empty(
        shape2, dtype=infinicore.float32, device=infinicore.device("cuda", 0)
    )

    t_gpu2.copy_(t_gpu)

    t_gpu2 = t_gpu2.permute([2, 0, 1]).contiguous()

    t_result = infinicore.from_blob(
        torch_tensor_result.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )

    t_result.copy_(t_gpu2)

    assert torch.equal(torch_tensor_ans, torch_tensor_result)
    print("Test passed")


if __name__ == "__main__":
    test()
