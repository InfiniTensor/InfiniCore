import infinicore
import torch


def test():
    shape = [2, 3, 4]
    torch_tensor_ans = torch.rand(shape, dtype=torch.float32, device="cpu")
    torch_tensor_result = torch.zeros(shape, dtype=torch.float32, device="cpu")

    t_cpu = infinicore.from_blob(
        torch_tensor_ans.data_ptr(),
        shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    t_gpu = t_cpu.to(infinicore.device("cuda", 0))

    t_gpu2 = infinicore.empty(
        shape, infinicore.float32, infinicore.device("cuda", 0), False
    )

    t_gpu2.copy_(t_gpu)

    t_result = infinicore.from_blob(
        torch_tensor_result.data_ptr(),
        shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    t_result.copy_(t_gpu2)

    assert torch.equal(torch_tensor_ans, torch_tensor_result)
    print("Test passed")


if __name__ == "__main__":
    test()
