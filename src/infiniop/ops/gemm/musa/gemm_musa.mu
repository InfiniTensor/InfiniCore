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
    std::shared_ptr<device::musa::Handle::Internal> &_internal,
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
    // support 2D and 3D tensor matmul

    // 1. set BatchMatMul operator Descriptor
    ::musa::dnn::BatchMatMul* matmul_operator = new ::musa::dnn::BatchMatMul();
    matmul_operator->SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR);

    // 2. set BatchMatMul Handle and stream  
    ::musa::dnn::Handle* mudnn_handles_t;
    mudnn_handles_t = new ::musa::dnn::Handle();
    mudnn_handles_t->SetStream((musaStream_t) stream);

    // 3. BatchMatMul Tensor create
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

    // 4. bind tensor addr
    out->SetAddr(c);
    left->SetAddr(a);
    right->SetAddr(b);

    // 5. BatchMatMul Tensor A and Tensor B compute config
    std::array<int64_t, 3> a_dims_array;
    std::array<int64_t, 3> a_stride_array;
    if (info.a_matrix.col_stride != 1){
        a_dims_array = {
            static_cast<int64_t>(info.batch),
            static_cast<int64_t>(info.k),
            static_cast<int64_t>(info.m)
        };

        a_stride_array = {
            static_cast<int64_t>(info.a_matrix.stride),
            static_cast<int64_t>(info.a_matrix.ld()),
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
            static_cast<int64_t>(info.a_matrix.stride),
            static_cast<int64_t>(info.a_matrix.ld()),
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
            static_cast<int64_t>(info.b_matrix.stride),
            static_cast<int64_t>(info.b_matrix.ld()),
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
            static_cast<int64_t>(info.b_matrix.stride),
            static_cast<int64_t>(info.b_matrix.ld()),
            static_cast<int64_t>(1)
        };
    }
    right->SetNdInfo(static_cast<int>(b_dims_array.size()), b_dims_array.data(), b_stride_array.data());

    // 6. muDNN BatchMatMul out only support row-major tensor
    std::array<int64_t, 3> c_dims_array;
    std::array<int64_t, 3> c_stride_array;
    c_dims_array = {
        static_cast<int64_t>(info.batch),
        static_cast<int64_t>(info.m),
        static_cast<int64_t>(info.n)
    };
    c_stride_array = {
        static_cast<int64_t>(info.c_matrix.stride),
        static_cast<int64_t>(info.c_matrix.ld()),
        static_cast<int64_t>(1)
    };

    out->SetNdInfo(static_cast<int>(c_dims_array.size()), c_dims_array.data(), c_stride_array.data());

    // 7. set BatchMatMul MemoryHandler
    ::musa::dnn::MemoryMaintainer maintainer = [](size_t size) -> ::musa::dnn::MemoryHandler {
        void* ptr = nullptr;
        musaMalloc(&ptr, size);  
        return ::musa::dnn::MemoryHandler(ptr, [](void* p) {
            if (p) musaFree(p); 
        });
    };

    // 8.  Tensor A and Tensor B transpose config
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

    // 9. set BatchMatMul GetWorkspaceSize
    size_t workspace_size_in_bytes = 0;
    matmul_operator->GetWorkspaceSize(*mudnn_handles_t, workspace_size_in_bytes, *out, *left, *right);


    // 10. set BatchMatMul Alpha Beta and Gamma
    matmul_operator->SetAlpha((double)alpha);
    matmul_operator->SetBeta((double)beta);
    matmul_operator->SetGamma(0.0);  

    // 11. BatchMatMul compute run
    matmul_operator->Run(
        *mudnn_handles_t,
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
    );

    // 12. delete 
    delete matmul_operator;
    delete mudnn_handles_t;
    delete out;
    delete left;
    delete right;

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
