import torch

import infinicore

from framework import (
    infinicore_tensor_from_torch,
    rearrange_tensor,
    to_infinicore_dtype,
    to_torch_dtype,
)


def comp(infini_result, torch_result):
    if infini_result.is_contiguous():
        print("comparing contiguous infini tensor")
    else:
        print("comparing non-contiguous infini tensor")

    if isinstance(infini_result, torch.Tensor):
        return torch.allclose(infini_result, torch_result, atol=1e-5, rtol=1e-5)

    # InfiniCore Tensor
    torch_result_from_infini = torch.zeros(
        infini_result.shape,
        dtype=to_torch_dtype(infini_result.dtype),
        device=infini_result.device.type,
    )

    temp_tensor = infinicore_tensor_from_torch(torch_result_from_infini)

    temp_tensor.copy_(infini_result)

    return torch.allclose(torch_result_from_infini, torch_result, atol=1e-5, rtol=1e-5)


def test():
    shape = [13, 4]
    strides = [10, 1]

    t_con = torch.rand(shape, dtype=torch.float32, device="cuda")
    t_clone = t_con.clone().detach()

    print("origin == torch clone   : ", comp(t_clone, t_con))

    t_str = rearrange_tensor(t_clone, strides)

    print("origin == torch strided : ", comp(t_str, t_con))

    i_con = i_str = infinicore.from_blob(
        t_clone.data_ptr(),
        shape,
        dtype=to_infinicore_dtype(t_str.dtype),
        device=infinicore.device("cuda", 0),
    )

    print("origin == infini conting: ", comp(i_con, t_con))

    i_str = infinicore.strided_from_blob(
        t_str.data_ptr(),
        shape,
        strides,
        dtype=to_infinicore_dtype(t_str.dtype),
        device=infinicore.device("cuda", 0),
    )

    print("origin == infini strided: ", comp(i_str, t_con))


if __name__ == "__main__":
    test()
