import torch

import infinicore

from framework import (
    create_strided_tensor_by_slicing,
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

    # print("origin == infini strided: ", comp(i_str, t_con))

    ori_t = create_strided_tensor_by_slicing(
        shape, strides, torch.float32, "cuda", mode="random"
    )
    cmp_t = create_strided_tensor_by_slicing(
        shape, strides, torch.float32, "cuda", mode="zeros"
    )
    dst_t = create_strided_tensor_by_slicing(
        shape, strides, torch.float32, "cuda", mode="zeros"
    )
    print("ori_t shape:", ori_t.shape, ", ori_t strides:", ori_t.stride())
    print("cmp_t shape:", cmp_t.shape, ", cmp_t strides:", cmp_t.stride())
    print("dst_t shape:", dst_t.shape, ", dst_t strides:", dst_t.stride())
    print("0: cmp_t == dst_t: ", torch.allclose(cmp_t, dst_t, atol=1e-5, rtol=1e-5))
    ori_i = infinicore.strided_from_blob(
        ori_t.data_ptr(),
        shape,
        strides,
        dtype=to_infinicore_dtype(ori_t.dtype),
        device=infinicore.device("cuda", 0),
    )
    print("ori_i shape:", ori_i.shape, ", ori_i strides:", ori_i.stride())
    cmp_i = infinicore.strided_from_blob(
        cmp_t.data_ptr(),
        shape,
        strides,
        dtype=to_infinicore_dtype(cmp_t.dtype),
        device=infinicore.device("cuda", 0),
    )
    print("1: cmp_t == dst_t: ", torch.allclose(cmp_t, dst_t, atol=1e-5, rtol=1e-5))
    dst_i = infinicore.strided_from_blob(
        dst_t.data_ptr(),
        shape,
        strides,
        dtype=to_infinicore_dtype(dst_t.dtype),
        device=infinicore.device("cuda", 0),
    )
    dst_i.copy_(ori_i)
    print("2: ori_t == dst_t: ", torch.allclose(ori_t, dst_t, atol=1e-5, rtol=1e-5))
    # ori_ci = ori_i.to(infinicore.device("cpu", 0))
    # dst_ci = dst_i.to(infinicore.device("cpu", 0))
    # # dst_ci.copy_(ori_ci)
    # ori_ct = create_strided_tensor_by_slicing(shape, strides, torch.float32, "cpu", mode="zeros")
    # dst_ct = create_strided_tensor_by_slicing(shape, strides, torch.float32, "cpu", mode="zeros")


if __name__ == "__main__":
    test()
