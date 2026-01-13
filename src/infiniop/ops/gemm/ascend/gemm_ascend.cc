#include "gemm_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_baddbmm.h>
#include <aclnnop/aclnn_batch_matmul.h>

#include <cstring>
#include <unordered_map>

// Custom hash function for alpha beta pair<float, float>
struct FloatPairHash {
    size_t operator()(const std::pair<float, float> &p) const {
        uint64_t combined;
        std::memcpy(reinterpret_cast<char *>(&combined), &p.first, sizeof(float));
        std::memcpy(reinterpret_cast<char *>(&combined) + sizeof(float), &p.second, sizeof(float));

        return std::hash<uint64_t>()(combined);
    }
};

struct FloatPairEqual {
    bool operator()(const std::pair<float, float> &a, const std::pair<float, float> &b) const {
        return a.first == b.first && a.second == b.second;
    }
};

namespace op::gemm::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c, a, b;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/aolapi/context/aclnnBaddbmm&aclnnInplaceBaddbmm.md
    int8_t mt;
    // alpha&beta hashmap
    std::unordered_map<std::pair<float, float>, aclOpExecutor *, FloatPairHash, FloatPairEqual> lookup;

    ~Opaque() {
        delete c;
        delete a;
        delete b;
        for (auto &item : lookup) {
            aclDestroyAclOpExecutor(item.second);
            GetRecentErrMsg();
        }
        lookup.clear();
    }
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
    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // Cast to aclTensor
    aclnnTensorDescriptor_t c, a, b;
    if (c_desc->ndim() == 3) {
        c = new aclnnTensorDescriptor(c_desc);
        a = new aclnnTensorDescriptor(a_desc);
        b = new aclnnTensorDescriptor(b_desc);
    } else if (c_desc->ndim() == 2) {
        c = new aclnnTensorDescriptor(
            toAclDataType(c_desc->dtype()),
            {1, static_cast<int64_t>(c_desc->dim(0)), static_cast<int64_t>(c_desc->dim(1))},
            {0, static_cast<int64_t>(c_desc->stride(0)), static_cast<int64_t>(c_desc->stride(1))},
            nullptr);
        a = new aclnnTensorDescriptor(
            toAclDataType(a_desc->dtype()),
            {1, static_cast<int64_t>(a_desc->dim(0)), static_cast<int64_t>(a_desc->dim(1))},
            {0, static_cast<int64_t>(a_desc->stride(0)), static_cast<int64_t>(a_desc->stride(1))},
            nullptr);
        b = new aclnnTensorDescriptor(
            toAclDataType(b_desc->dtype()),
            {1, static_cast<int64_t>(b_desc->dim(0)), static_cast<int64_t>(b_desc->dim(1))},
            {0, static_cast<int64_t>(b_desc->stride(0)), static_cast<int64_t>(b_desc->stride(1))},
            nullptr);
    } else {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto tc = c->tensor,
         ta = a->tensor,
         tb = b->tensor;

    std::unordered_map<std::pair<float, float>, aclOpExecutor *, FloatPairHash, FloatPairEqual> lookup;
    aclOpExecutor *executor = nullptr;
    size_t workspace_size = 0;
    int8_t mt = 1;
    // float alpha_val = 0.5f;
    // float beta_val = 0.5f;
    // aclScalar *alpha = aclCreateScalar(&alpha_val, aclDataType::ACL_FLOAT);
    // aclScalar *beta = aclCreateScalar(&beta_val, aclDataType::ACL_FLOAT);
    std::cout << c->toString() << std::endl;
    std::cout << a->toString() << std::endl;
    std::cout << b->toString() << std::endl;
    // printf("alpha: %f, beta: %f\n", alpha_val, beta_val);
    // CHECK_ACL(aclnnInplaceBaddbmmGetWorkspaceSize(tc, ta, tb, beta, alpha, mt, &workspace_size, &executor));
    // CHECK_ACL(aclnnBaddbmmGetWorkspaceSize(tc, ta, tb, beta, alpha, tc, mt, &workspace_size, &executor));
    CHECK_ACL(aclnnBatchMatMulGetWorkspaceSize(ta, tb, tc, mt, &workspace_size, &executor));
    printf("Workspace size in Gemm op kernel: %zu\n", workspace_size);
    // GetRecentErrMsg();
    // CHECK_ACL(aclDestroyAclOpExecutor(executor));
    // GetRecentErrMsg();
    // GetRecentErrMsg();
    // CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
    // GetRecentErrMsg();
    // lookup[std::make_pair(alpha_val, beta_val)] = executor;

    *desc_ptr = new Descriptor(
        dtype, workspace_size,
        new Opaque{
            c,
            a,
            b,
            mt,
            std::move(lookup)},
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspaceSize_,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    auto tc = _opaque->c->tensor,
         ta = _opaque->a->tensor,
         tb = _opaque->b->tensor;

    size_t workspace_size = _workspace_size;
    aclOpExecutor *executor;
    auto key = std::make_pair(alpha, beta);
    if (_opaque->lookup.find(key) != _opaque->lookup.end()) {
        executor = _opaque->lookup[key];
    } else {
        // aclScalar *alpha_ = aclCreateScalar(&alpha, aclDataType::ACL_FLOAT);
        // aclScalar *beta_ = aclCreateScalar(&beta, aclDataType::ACL_FLOAT);
        // CHECK_ACL(aclnnInplaceBaddbmmGetWorkspaceSize(tc, ta, tb, beta_, alpha_, _opaque->mt, &workspace_size, &executor));
        // CHECK_ACL(aclnnBaddbmmGetWorkspaceSize(tc, ta, tb, beta_, alpha_, tc, _opaque->mt, &workspace_size, &executor));
        CHECK_ACL(aclnnBatchMatMulGetWorkspaceSize(ta, tb, tc, _opaque->mt, &workspace_size, &executor));
        GetRecentErrMsg();
        CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));
        GetRecentErrMsg();
        _opaque->lookup[key] = executor;
    }

    if (workspaceSize_ < workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    // CHECK_ACL(aclSetTensorAddr(executor, 0, tc, c));
    // CHECK_ACL(aclSetTensorAddr(executor, 1, ta, (void *)a));
    // CHECK_ACL(aclSetTensorAddr(executor, 2, tb, (void *)b));
    // CHECK_ACL(aclSetTensorAddr(executor, 3, tc, (void *)c));
    CHECK_ACL(aclSetTensorAddr(executor, 0, ta, (void *)a));
    CHECK_ACL(aclSetTensorAddr(executor, 1, tb, (void *)b));
    CHECK_ACL(aclSetTensorAddr(executor, 2, tc, (void *)c));
    GetRecentErrMsg();
    // CHECK_ACL(aclnnBaddbmm(workspace, workspace_size, executor, stream));
    CHECK_ACL(aclnnBatchMatMul(workspace, workspace_size, executor, stream));
    GetRecentErrMsg();
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::gemm::ascend
