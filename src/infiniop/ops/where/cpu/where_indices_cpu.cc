#include "where_indices_cpu.h"
#include <cstring>
#include <functional>
#include <vector>

namespace op::where::cpu {

infiniStatus_t IndicesDescriptor::create(
    infiniopHandle_t handle_,
    IndicesDescriptor **desc_ptr,
    infiniopTensorDescriptor_t cond_desc) {

    // 检查条件必须是 bool 类型
    if (cond_desc->dtype() != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t numel = cond_desc->numel();
    int ndim = static_cast<int>(cond_desc->ndim());

    std::vector<size_t> shape(ndim);
    std::vector<ptrdiff_t> strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = cond_desc->shape()[i];
        strides[i] = cond_desc->stride(i);
    }

    *desc_ptr = new IndicesDescriptor(
        numel, ndim, shape.data(), strides.data(),
        INFINI_DEVICE_CPU, 0);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t IndicesDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void **outputs,
    const void *cond,
    void *stream,
    size_t *num_true) const {

    const bool *cond_ptr = reinterpret_cast<const bool *>(cond);
    int64_t **output_ptrs = new int64_t *[_ndim];
    for (int i = 0; i < _ndim; ++i) {
        output_ptrs[i] = reinterpret_cast<int64_t *>(outputs[i]);
    }

    // 使用递归函数遍历所有多维索引，正确处理 strided tensor
    std::vector<size_t> indices(_ndim, 0);
    size_t output_idx = 0;

    // 递归函数来遍历所有多维索引
    std::function<void(int)> traverse = [&](int dim) {
        if (dim == _ndim) {
            // 计算内存偏移（考虑 stride）
            size_t offset = 0;
            for (int i = 0; i < _ndim; ++i) {
                offset += indices[i] * static_cast<size_t>(_strides[i]);
            }

            // 检查条件是否为 True
            if (cond_ptr[offset]) {
                // 记录多维索引
                for (int i = 0; i < _ndim; ++i) {
                    output_ptrs[i][output_idx] = static_cast<int64_t>(indices[i]);
                }
                output_idx++;
            }
        } else {
            // 递归遍历当前维度的所有可能值
            for (size_t i = 0; i < _shape[dim]; ++i) {
                indices[dim] = i;
                traverse(dim + 1);
            }
        }
    };

    traverse(0);

    *num_true = output_idx;
    delete[] output_ptrs;
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::where::cpu
