#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "gemm_musa.h"
#include <mudnn.h>

namespace op::gemm::musa {

struct Descriptor::Opaque {
    std::shared_ptr<device::musa::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::musa::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculate(
    const MatmulInfo &info,
    std::shared_ptr<device::musa::Handle::Internal> &_internal, // _internal 是关键
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) {

    // 0. For muDNN development, refer to the official documentation and the following headers:
    // - /usr/local/musa/include/mudnn_base.h
    // - /usr/local/musa/include/mudnn_math.h
    // - /usr/local/musa/include/mudnn.h
    // only support 2D and 3D tensor matmul

    // 使用 _internal->useMudnn 来获取和管理 mudnn 句柄
    CHECK_STATUS(_internal->useMudnn(
        (musaStream_t)stream, // 将 stream 传递给 useMudnn
        [&](::musa::dnn::Handle &handle) { // mudnn 句柄通过引用传递进来，无需手动 new/delete
            // 1. set BatchMatMul operator Descriptor
            // 注意：BatchMatMul operator Descriptor 仍然需要每次创建，因为它不属于 handle 的一部分
            ::musa::dnn::BatchMatMul* matmul_operator = new ::musa::dnn::BatchMatMul();
            matmul_operator->SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR);

            // 2. set BatchMatMul Handle and stream
            // mudnn_handles_t 现在就是 useMudnn 传进来的 'handle'
            // handle.SetStream((musaStream_t) stream); // 这一行现在由 useMudnn 内部完成，无需重复设置

            // 3. BatchMatMul Tensor config
            // Tensor 对象仍然需要每次创建，因为它们与特定的输入/输出形状和数据类型相关
            ::musa::dnn::Tensor *out = new ::musa::dnn::Tensor();
            ::musa::dnn::Tensor *left = new ::musa::dnn::Tensor();
            ::musa::dnn::Tensor *right = new ::musa::dnn::Tensor();

            if constexpr (std::is_same<Tdata, half>::value) {
                out->SetType(::musa::dnn::Tensor::Type::HALF);
                left->SetType(::musa::dnn::Tensor::Type::HALF);
                right->SetType(::musa::dnn::Tensor::Type::HALF);
            }
            else {
                out->SetType(::musa::dnn::Tensor::Type::FLOAT);
                left->SetType(::musa::dnn::Tensor::Type::FLOAT);
                right->SetType(::musa::dnn::Tensor::Type::FLOAT);
            }

            // 5. bind tensor addr
            out->SetAddr(c);
            left->SetAddr(a);
            right->SetAddr(b);

            // 4. BatchMatMul Tensor compute config (与原代码相同)
            std::array<int64_t, 3> a_dims_array;
            std::array<int64_t, 3> a_stride_array;
            if (info.a_matrix.col_stride != 1){
                a_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.k),
                    static_cast<int64_t>(info.m)
                };

                a_stride_array = {
                    static_cast<int64_t>(info.m * info.k),
                    static_cast<int64_t>(info.m),
                    static_cast<int64_t>(1)
                };
            }
            else{
                a_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.m),
                    static_cast<int64_t>(info.k)
                };
                a_stride_array = {
                    static_cast<int64_t>(info.m * info.k),
                    static_cast<int64_t>(info.k),
                    static_cast<int64_t>(1)
                };
            }
            left->SetNdInfo(static_cast<int>(a_dims_array.size()), a_dims_array.data(), a_stride_array.data());


            std::array<int64_t, 3> b_dims_array;
            std::array<int64_t, 3> b_stride_array;
            if (info.b_matrix.col_stride != 1){
                b_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.n),
                    static_cast<int64_t>(info.k)
                };

                b_stride_array = {
                    static_cast<int64_t>(info.n * info.k),
                    static_cast<int64_t>(info.k),
                    static_cast<int64_t>(1)
                };
            }
            else{
                b_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.k),
                    static_cast<int64_t>(info.n)
                };
                b_stride_array = {
                    static_cast<int64_t>(info.n * info.k),
                    static_cast<int64_t>(info.n),
                    static_cast<int64_t>(1)
                };
            }
            right->SetNdInfo(static_cast<int>(b_dims_array.size()), b_dims_array.data(), b_stride_array.data());

            std::array<int64_t, 3> c_dims_array;
            std::array<int64_t, 3> c_stride_array;
            if (info.c_matrix.col_stride != 1){
                c_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.n),
                    static_cast<int64_t>(info.m)
                };

                c_stride_array = {
                    static_cast<int64_t>(info.m * info.n),
                    static_cast<int64_t>(info.m),
                    static_cast<int64_t>(1)
                };
            }
            else{
                c_dims_array = {
                    static_cast<int64_t>(info.batch),
                    static_cast<int64_t>(info.m),
                    static_cast<int64_t>(info.n)
                };
                c_stride_array = {
                    static_cast<int64_t>(info.m * info.n),
                    static_cast<int64_t>(info.n),
                    static_cast<int64_t>(1)
                };
            }
            out->SetNdInfo(static_cast<int>(c_dims_array.size()), c_dims_array.data(), c_stride_array.data());

            // 6. set BatchMatMul MemoryHandler
            // MemoryMaintainer 可以在 lambda 外部定义一次，或者像这样每次定义，只要其内部的 musaMalloc/Free 是线程安全的，并与当前设备兼容即可。
            // 由于 musaMalloc/Free 默认作用于当前设备，并且 useMudnn 应该保证了 handle 与当前设备的匹配，所以这里通常没问题。
            ::musa::dnn::MemoryMaintainer maintainer = [](size_t size) -> ::musa::dnn::MemoryHandler {
                void* ptr = nullptr;
                musaMalloc(&ptr, size);
                return ::musa::dnn::MemoryHandler(ptr, [](void* p) {
                    if (p) musaFree(p);
                });
            };

            if(info.a_matrix.col_stride == 1 && info.b_matrix.col_stride !=1){
                matmul_operator->SetTranspose(false, true);
            }
            else if(info.a_matrix.col_stride != 1 && info.b_matrix.col_stride ==1){
                matmul_operator->SetTranspose(true, false);
            }
            else if(info.a_matrix.col_stride != 1 && info.b_matrix.col_stride !=1){
                matmul_operator->SetTranspose(true, true);
            }
            else{
                matmul_operator->SetTranspose(false, false);
            }

            // 7. set BatchMatMul GetWorkspaceSize
            size_t workspace_size_in_bytes = 0;
            // 传入 useMudnn 提供的 'handle'
            CHECK_MUDNN(matmul_operator->GetWorkspaceSize(handle, workspace_size_in_bytes, *out, *left, *right));


            // 8. set BatchMatMul Alpha Beta and Gamma
            matmul_operator->SetAlpha((double)alpha);
            matmul_operator->SetBeta((double)beta);
            matmul_operator->SetGamma(0.0);

            // 9. BatchMatMul compute run
            // 传入 useMudnn 提供的 'handle'
            CHECK_MUDNN(matmul_operator->Run(
                handle, // 使用传进来的 handle
                *out,
                *left,
                *right,
                static_cast<int64_t>(info.batch),
                static_cast<int64_t>(info.m),
                static_cast<int64_t>(info.n),
                static_cast<int64_t>(info.k),
                static_cast<int64_t>(info.a_matrix.ld()),
                static_cast<int64_t>(info.b_matrix.ld()),
                static_cast<int64_t>(info.c_matrix.ld()),
                static_cast<int64_t>(info.a_matrix.stride),
                static_cast<int64_t>(info.b_matrix.stride),
                static_cast<int64_t>(info.c_matrix.stride),
                maintainer
            ));

            // 10. delete 仅删除在 lambda 内部 new 的对象
            delete matmul_operator;
            delete out;
            delete left;
            delete right;

            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     float beta,
                                     const void *a,
                                     const void *b,
                                     float alpha,
                                     void *stream) const {
    switch (_dtype) {
        case INFINI_DTYPE_F16:
            return musa::calculate<half>(_info, _opaque->internal, c, beta, a, b, alpha, stream);
        case INFINI_DTYPE_F32:
            return musa::calculate<float>(_info,_opaque->internal, c, beta, a, b, alpha, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::gemm::musa
