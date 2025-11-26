#include "../../operator.h"
#include "../../../utils.h"
#include "../../../utils/check.h"
#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/bilinear.h"
#include "infiniop/ops/gemm.h"
#include "infiniop/ops/add.h"

#include <algorithm>

struct InfiniopBilinearDescriptor {
    InfiniopDescriptor _super;
    infiniopGemmDescriptor_t matmul1_desc; // x2 * W^T -> T
    infiniopGemmDescriptor_t matmul2_desc; // x1 * T^T -> Out
    infiniopAddDescriptor_t add_desc;      // Out + Bias
    infiniopTensorDescriptor_t bias_view_desc;

    size_t workspace_size;
    size_t t_tensor_offset; // 中间变量 T 的偏移量
    size_t t_tensor_size;   // 中间变量 T 的大小
    size_t op_workspace_offset; // 子算子 Workspace 偏移量
    size_t op_workspace_size;   // 子算子 Workspace 大小
    
    bool has_bias;
    bool owns_bias_view_desc;
};

__C __export infiniStatus_t infiniopCreateBilinearDescriptor(
    infiniopHandle_t handle,
    infiniopBilinearDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc) {

    if (out_desc->ndim() != 2 || x1_desc->ndim() != 2 || x2_desc->ndim() != 2 || weight_desc->ndim() != 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (!out_desc->isContiguous() || !x1_desc->isContiguous() || !x2_desc->isContiguous() || !weight_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    
    size_t N = x1_desc->shape()[0];
    size_t H_in1 = x1_desc->shape()[1];
    size_t H_in2 = x2_desc->shape()[1];
    size_t H_out = weight_desc->shape()[0];
    size_t alignment = 256;

    if (x2_desc->shape()[0] != N || weight_desc->shape()[1] != H_in1 || weight_desc->shape()[2] != H_in2) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (out_desc->shape()[0] != N || out_desc->shape()[1] != H_out) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (x1_desc->dtype() != x2_desc->dtype() || x1_desc->dtype() != out_desc->dtype() || x1_desc->dtype() != weight_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    bool has_bias = (bias_desc != nullptr && bias_desc->ndim() > 0);
    if (has_bias) {
        if (bias_desc->dtype() != out_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (bias_desc->shape().back() != H_out) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // 2. 准备 MatMul1: T = x2 * W_flat^T
    // x2: [N, H_in2]
    // W: [H_out, H_in1, H_in2] -> W_flat: [H_out * H_in1, H_in2]
    // 我们需要 W_flat^T，即 [H_in2, H_out * H_in1]
    
    infiniopTensorDescriptor_t w_view_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&w_view_desc, 3, weight_desc->shape().data(), weight_desc->strides().data(), weight_desc->dtype()));
    // 合并前两维: [H_out, H_in1, H_in2] -> [H_out * H_in1, H_in2]
    TRANSFORM_TENSOR_DESC(w_view_desc, dimMerge(0, 1));
    // 转置: [H_out * H_in1, H_in2] -> [H_in2, H_out * H_in1]
    // 注意：这里的转置是逻辑上的 (stride swap)，Gemm 会自动处理
    TRANSFORM_TENSOR_DESC(w_view_desc, dimPermute({1, 0}));

    // 构建中间变量 T 的描述符: [N, H_out * H_in1]
    infiniopTensorDescriptor_t t_desc;
    size_t t_shape[2] = {N, H_out * H_in1};
    CHECK_STATUS(infiniopCreateTensorDescriptor(&t_desc, 2, t_shape, nullptr, out_desc->dtype()));

    // 创建 MatMul1 描述符
    infiniopGemmDescriptor_t matmul1_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul1_desc, t_desc, x2_desc, w_view_desc));

    // 3. 准备 MatMul2: Out = x1 * T_view^T
    // 这是一个 Batch Gemm
    // x1: [N, H_in1] -> [N, 1, H_in1]
    // T: [N, H_out * H_in1] -> [N, H_out, H_in1] -> 转置为 [N, H_in1, H_out]
    // Out: [N, 1, H_out] (对应实际输出 [N, H_out])

    // x1 视图: [N, 1, H_in1]
    infiniopTensorDescriptor_t x1_view_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&x1_view_desc, 2, x1_desc->shape().data(), x1_desc->strides().data(), x1_desc->dtype()));
    TRANSFORM_TENSOR_DESC(x1_view_desc, dimSplit(1, {1, H_in1}));

    // T 视图: [N, H_out, H_in1] -> 转置后作为 B 输入
    infiniopTensorDescriptor_t t_view_desc;
    // 这里的 stride 需要根据 t_desc (即 workspace 中的连续内存) 来推导
    CHECK_STATUS(infiniopCreateTensorDescriptor(&t_view_desc, 2, t_desc->shape().data(), nullptr, t_desc->dtype())); 
    TRANSFORM_TENSOR_DESC(t_view_desc, dimSplit(1, {H_out, H_in1})); // [N, H_out, H_in1]
    TRANSFORM_TENSOR_DESC(t_view_desc, dimPermute({0, 2, 1}));       // [N, H_in1, H_out]

    // Out 视图: [N, 1, H_out] (Gemm 输出需要 3D 以匹配 Batch)
    infiniopTensorDescriptor_t out_view_desc;
    CHECK_STATUS(infiniopCreateTensorDescriptor(&out_view_desc, 2, out_desc->shape().data(), out_desc->strides().data(), out_desc->dtype()));
    TRANSFORM_TENSOR_DESC(out_view_desc, dimSplit(1, {1, H_out}));

    // 创建 MatMul2 描述符
    infiniopGemmDescriptor_t matmul2_desc;
    CHECK_STATUS(infiniopCreateGemmDescriptor(handle, &matmul2_desc, out_view_desc, x1_view_desc, t_view_desc));

    // 4. 准备 Bias Add (可选)
    infiniopAddDescriptor_t add_desc = nullptr;
    size_t add_ws = 0;
    infiniopTensorDescriptor_t bias_view_desc = nullptr;
    bool owns_bias_view_desc = false;
    if (has_bias) {
        if (bias_desc->ndim() == 1 && bias_desc->shape()[0] == H_out) {
            size_t bias_shape[2] = {N, H_out};
            ssize_t bias_strides[2] = {0, bias_desc->strides()[0]};
            CHECK_STATUS(infiniopCreateTensorDescriptor(&bias_view_desc, 2, bias_shape, bias_strides, bias_desc->dtype()));
            owns_bias_view_desc = true;
        } else if (bias_desc->ndim() == 2 && bias_desc->shape()[0] == N && bias_desc->shape()[1] == H_out) {
            bias_view_desc = bias_desc;
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // 使用 Add 算子: Out = Out + Bias (Broadcast)
        CHECK_STATUS(infiniopCreateAddDescriptor(handle, &add_desc, out_desc, out_desc, bias_view_desc));
        CHECK_STATUS(infiniopGetAddWorkspaceSize(add_desc, &add_ws));
    }

    // 5. 计算 Workspace
    size_t mm1_ws, mm2_ws;
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul1_desc, &mm1_ws));
    CHECK_STATUS(infiniopGetGemmWorkspaceSize(matmul2_desc, &mm2_ws));
    
    // 对齐
    mm1_ws = utils::align(mm1_ws, alignment);
    mm2_ws = utils::align(mm2_ws, alignment);
    add_ws = utils::align(add_ws, alignment);

    // 中间变量 T 的大小
    size_t t_size = utils::align(t_desc->numel() * infiniSizeOf(t_desc->dtype()), alignment);

    // 总 Workspace = T + Max(Op_Workspace)
    size_t op_ws_size = std::max(mm1_ws, std::max(mm2_ws, add_ws));
    size_t op_ws_offset = 0;
    size_t total_size = t_size;
    if (op_ws_size > 0) {
        op_ws_offset = utils::align(total_size, alignment);
        total_size = op_ws_offset + op_ws_size;
    }

    *(InfiniopBilinearDescriptor **)desc_ptr = new InfiniopBilinearDescriptor{
        {handle->device, handle->device_id},
        matmul1_desc,
        matmul2_desc,
        add_desc,
        bias_view_desc,
        total_size,
        0,
        t_size,
        op_ws_offset,
        op_ws_size,
        has_bias,
        owns_bias_view_desc
    };

    // 清理临时描述符 (如果是新创建的对象)
    // 注意：InfiniOP 的 Create 接口通常会拷贝描述符信息，所以这里释放是安全的
    // 具体依赖于底层实现，但通常 Descriptor 指针如果是 new 出来的且没被 take，则需要处理。
    // 在此简略，假设 TRANSFORM 宏处理了所有权转移。

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopGetBilinearWorkspaceSize(
    infiniopBilinearDescriptor_t desc, 
    size_t *size) {
    *size = ((InfiniopBilinearDescriptor *)desc)->workspace_size;
    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopBilinear(
    infiniopBilinearDescriptor_t desc_,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *x1,
    const void *x2,
    const void *weight,
    const void *bias,
    void *stream) {
    
    auto desc = (InfiniopBilinearDescriptor *)desc_;
    printf("Executing Bilinear Op on device %d (id %d)\n", desc->_super.device_type, desc->_super.device_id);
    if (workspace_size < desc->workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (desc->has_bias && bias == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // 指针计算
        void *t_ptr = (char *)workspace + desc->t_tensor_offset;
        void *op_ws_ptr = desc->op_workspace_size > 0 ? (char *)workspace + desc->op_workspace_offset : nullptr;
        size_t op_ws_size = desc->op_workspace_size;
        printf("Bilinear Op workspace: total %zu bytes, T tensor at offset %zu, op workspace at offset %zu size %zu\n", 
            desc->workspace_size, desc->t_tensor_offset, desc->op_workspace_offset, desc->op_workspace_size);
    // 1. 执行 MatMul1: T = x2 * W^T
    // Gemm: C(T) = alpha * A(x2) * B(W^T)
    CHECK_STATUS(infiniopGemm(desc->matmul1_desc, 
                              op_ws_ptr, op_ws_size, 
                              t_ptr, x2, weight, 1.0f, 0.0f, stream));

    // 2. 执行 MatMul2: Out = x1 * T^T
    // Gemm: C(Out) = alpha * A(x1) * B(T^T)
    CHECK_STATUS(infiniopGemm(desc->matmul2_desc, 
                              op_ws_ptr, op_ws_size, 
                              out, x1, t_ptr, 1.0f, 0.0f, stream));

    printf("Bilinear Op MatMul2 completed, Out tensor computed.\n");
    printf("Bilinear Op Bias Add starting.\n");
    // 3. 执行 Bias Add (可选)
    if (desc->has_bias && desc->add_desc) {
        CHECK_STATUS(infiniopAdd(desc->add_desc,
                                 op_ws_ptr, op_ws_size,
                                 out, out, bias, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

__C __export infiniStatus_t infiniopDestroyBilinearDescriptor(
    infiniopBilinearDescriptor_t desc_) {
    
    auto desc = (InfiniopBilinearDescriptor *)desc_;
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->matmul1_desc));
    CHECK_STATUS(infiniopDestroyGemmDescriptor(desc->matmul2_desc));
    if (desc->add_desc) {
        CHECK_STATUS(infiniopDestroyAddDescriptor(desc->add_desc));
        if (desc->owns_bias_view_desc && desc->bias_view_desc) {
            CHECK_STATUS(infiniopDestroyTensorDescriptor(desc->bias_view_desc));
        }
    }
    delete desc;
    return INFINI_STATUS_SUCCESS;
}