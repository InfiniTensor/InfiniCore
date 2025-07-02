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

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
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

    musaDataType a_type, b_type, c_type;
    mublasComputeType_t compute_type;
    Tdata alpha_, beta_;

    if constexpr (std::is_same<Tdata, half>::value) {
        alpha_ = __float2half(alpha);
        beta_ = __float2half(beta);
        a_type = b_type = c_type = MUSA_R_16F;
        compute_type = MUBLAS_COMPUTE_16F;
    } else {
        alpha_ = alpha;
        beta_ = beta;
        a_type = b_type = c_type = MUSA_R_32F;
        compute_type = MUBLAS_COMPUTE_32F_FAST_TF32;
    }

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = info.a_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? MUBLAS_OP_N : MUBLAS_OP_T;


    // 0. For muDNN development, refer to the official documentation and the following headers:
    // - /usr/local/musa/include/mudnn_base.h
    // - /usr/local/musa/include/mudnn_math.h
    // - /usr/local/musa/include/mudnn.h
    // only support 3D tensor matmul

    // 1. set BatchMatMul operator Descriptor
    ::musa::dnn::BatchMatMul* matmul_operator = new ::musa::dnn::BatchMatMul();
    matmul_operator->SetComputeMode(::musa::dnn::BatchMatMul::ComputeMode::TENSOR);

    // 2. set BatchMatMul Handle and stream  
    ::musa::dnn::Handle* mudnn_handles_t;
    mudnn_handles_t = new ::musa::dnn::Handle();
    mudnn_handles_t->SetStream((musaStream_t) stream);

    // 3. BatchMatMul Tensor config
    ::musa::dnn::Tensor *out = new ::musa::dnn::Tensor();
    ::musa::dnn::Tensor *left = new ::musa::dnn::Tensor();
    ::musa::dnn::Tensor *right = new ::musa::dnn::Tensor();

    out->SetType(::musa::dnn::Tensor::Type::FLOAT);
    left->SetType(::musa::dnn::Tensor::Type::FLOAT);
    right->SetType(::musa::dnn::Tensor::Type::FLOAT);

    // std::cout << "info.batch: " << info.batch << std::endl;
    // std::cout << "info.m: " << info.m << std::endl;
    // std::cout << "info.n: " << info.n << std::endl;
    // std::cout << "info.k: " << info.k << std::endl;

    int64_t a_dims[3];
    a_dims[0] = info.batch;
    a_dims[1] = info.m;
    a_dims[2] = info.k;
    left->SetNdInfo(3, a_dims);
    ::musa::dnn::Status status1 = left->SetNdInfo(3, a_dims);
    // if (status1 == ::musa::dnn::Status::SUCCESS) {
    //     std::cerr << "Success to set left." << std::endl;
    // }

    int64_t b_dims[3];
    b_dims[0] = info.batch;
    b_dims[1] = info.k;
    b_dims[2] = info.n;
    right->SetNdInfo(3, b_dims);

    int64_t c_dims[3];
    c_dims[0] = info.batch;
    c_dims[1] = info.m;
    c_dims[2] = info.n;
    out->SetNdInfo(3, c_dims);

    out->SetAddr(c);
    left->SetAddr(a);
    right->SetAddr(b);


    // 4. set BatchMatMul MemoryHandler
    ::musa::dnn::MemoryMaintainer maintainer = [](size_t size) -> ::musa::dnn::MemoryHandler {
        void* ptr = nullptr;
        musaMalloc(&ptr, size);  
        return ::musa::dnn::MemoryHandler(ptr, [](void* p) {
            if (p) musaFree(p); 
        });
    };

    // 5. set BatchMatMul GetWorkspaceSize
    size_t workspace_size_in_bytes = 0;
    matmul_operator->GetWorkspaceSize(*mudnn_handles_t, workspace_size_in_bytes, *out, *left, *right);

    // std::cout << "info.is_transed: " << info.is_transed << std::endl;
    // std::cout << "left dims: " << a_dims[0] << ", " << a_dims[1] << ", " << a_dims[2] << std::endl;
    // std::cout << "right dims: " << b_dims[0] << ", " << b_dims[1] << ", " << b_dims[2] << std::endl;
    // std::cout << "output dims: " << c_dims[0] << ", " << c_dims[1] << ", " << c_dims[2] << std::endl;


    if (info.is_transed) {
        matmul_operator->SetTranspose(true, true);
    } else {
        matmul_operator->SetTranspose(false, false);
    }

    // 6. set BatchMatMul Alpha Beta and Gamma
    matmul_operator->SetAlpha((double)alpha);
    matmul_operator->SetBeta((double)beta);
    matmul_operator->SetGamma(0.0);  

    matmul_operator->Run(
        *mudnn_handles_t,
        *out,
        *left,
        *right,
        maintainer
    );

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
