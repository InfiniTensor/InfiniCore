print("--- [PY_DEBUG] Script started ---")

import ctypes
import torch
# 【修正】从 libinfiniop 中导入所有需要的类型和枚举
from libinfiniop import (
    LIBINFINIOP,
    check_error,
    InfiniDtype,
    InfiniDeviceEnum,
    infiniopHandle_t, # <<<<<< 导入 Handle 类型
    infiniopTensorDescriptor_t, # <<<<<< 导入 TensorDescriptor 类型
    infiniopOperatorDescriptor_t
)

print("--- [PY_DEBUG] Imports successful ---")

# 【修正】使用预定义的 infiniopHandle_t 类型
handle = infiniopHandle_t()
check_error(LIBINFINIOP.infiniopCreateHandle(ctypes.byref(handle), InfiniDeviceEnum.CPU, 0))
print("--- [PY_DEBUG] Handle created ---")

# 手动创建 Tensor Descriptors
shape = (5, 3)
dim = 1
index_len = 3

input_shape = list(shape)
input_shape[dim] = index_len

# 【修正】使用预定义的 infiniopTensorDescriptor_t 类型
output_desc = infiniopTensorDescriptor_t()
input_desc = infiniopTensorDescriptor_t()
index_desc = infiniopTensorDescriptor_t()

check_error(LIBINFINIOP.infiniopCreateTensorDescriptor(
    ctypes.byref(output_desc), len(shape),
    (ctypes.c_size_t * len(shape))(*shape), None, InfiniDtype.F32
))
check_error(LIBINFINIOP.infiniopCreateTensorDescriptor(
    ctypes.byref(input_desc), len(input_shape),
    (ctypes.c_size_t * len(input_shape))(*input_shape), None, InfiniDtype.F32
))
check_error(LIBINFINIOP.infiniopCreateTensorDescriptor(
    ctypes.byref(index_desc), 1,
    (ctypes.c_size_t * 1)(index_len), None, InfiniDtype.I64
))
print("--- [PY_DEBUG] Tensor descriptors created ---")

# 调用我们自己的 C-API
op_descriptor = infiniopOperatorDescriptor_t()

print("--- [PY_DEBUG] About to call infiniopCreateIndexCopyInplaceDescriptor ---")

check_error(
    LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
        handle, # <<<<<< 现在 handle 的类型是正确的
        ctypes.byref(op_descriptor),
        input_desc,
        output_desc,
        dim,
        index_desc,
    )
)

print("--- [PY_DEBUG] infiniopCreateIndexCopyInplaceDescriptor SUCCESSFUL! ---")

# 清理资源
# 【修正】销毁 op_descriptor 需要使用我们自己的 C-API
check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(op_descriptor))
check_error(LIBINFINIOP.infiniopDestroyTensorDescriptor(output_desc))
check_error(LIBINFINIOP.infiniopDestroyTensorDescriptor(input_desc))
check_error(LIBINFINIOP.infiniopDestroyTensorDescriptor(index_desc))
check_error(LIBINFINIOP.infiniopDestroyHandle(handle))

print("--- [PY_DEBUG] Script finished successfully! ---")