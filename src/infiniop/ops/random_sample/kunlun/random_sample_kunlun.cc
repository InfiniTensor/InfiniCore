#include "random_sample_kunlun.h"
#include "../../../devices/kunlun/kunlun_common.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include "../info.h"
#include <assert.h>
void sample_I64(void *result, float *destination, int *topk_indices, float random_val,
                float topp,
                int topk_, XPUStream stream);
void sample_I32(void *result, float *destination, int *topk_indices, float random_val,
                float topp,
                int topk_, XPUStream stream);

namespace op::random_sample::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    size_t workspace_size = 3 * probs_desc->numel() * infiniSizeOf(probs_desc->dtype()) + probs_desc->numel() * infiniSizeOf(infiniDtype_t::INFINI_DTYPE_I32);

    *desc_ptr = new Descriptor(
        info,
        workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

infiniStatus_t random_sample_kernel(void *workspace,
                                    size_t workspace_size,
                                    std::shared_ptr<device::kunlun::Handle::Internal> internal,
                                    infiniDtype_t dt_p,
                                    infiniDtype_t dt_i,
                                    void *result,
                                    const void *probs,
                                    float random_val,
                                    float topp,
                                    int topk,
                                    float temperature,
                                    int64_t n,
                                    void *stream) {
    int topk_ = topk <= (int)n ? topk : (int)n;
    bool dosample = topk_ > 1 && temperature != 0.0f && topp != 0.0f && random_val != 0.0f;
    char *workspace_value = reinterpret_cast<char *>(workspace);

    if (dosample) {
        float *topk_values = (float *)workspace_value; //(topk_, )
        float *probs_F32 = topk_values + topk_;        //(n, )
        float *destination = probs_F32 + n;            //(n, )
        char *workspace_index = workspace_value + (2 * n + topk_) * sizeof(float);
        int *topk_indices = (int *)workspace_index; //(topk_)

        switch (dt_p) {
        case INFINI_DTYPE_F16:
            CHECK_STATUS(internal->useXdnn(
                (kunlunStream_t)stream,
                [&](xdnnHandle_t handle) {
                    CHECK_KUNLUN((xdnn::cast<float16, float>(handle, (float16 *)probs, probs_F32, n)));
                    CHECK_KUNLUN((xdnn::sorted_topk<float>(handle, probs_F32, topk_values, topk_indices, 1, n, topk_, true, true)));
                    float max_value = 0.0f;
                    xpu_memcpy(&max_value, topk_values, sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
                    CHECK_KUNLUN((xdnn::add_scalar<float>(handle, probs_F32, destination, max_value, -1.0f, n)));
                    CHECK_KUNLUN((xdnn::mul_scalar<float>(handle, destination, destination, 1.0 / temperature, n)));
                    CHECK_KUNLUN((xdnn::softmax<float>(handle, destination, destination, {n}, 0)));
                    CHECK_KUNLUN((xdnn::cumsum<float>(handle, destination, destination, {n}, false, false, 0)));
                    return INFINI_STATUS_SUCCESS;
                }));

            if (dt_i == INFINI_DTYPE_I64) {
                sample_I64(result, destination, topk_indices, random_val,
                           topp,
                           topk_, reinterpret_cast<kunlunStream_t>(stream));
                return INFINI_STATUS_SUCCESS;
            } else if (dt_i == INFINI_DTYPE_I32) {
                sample_I32(result, destination, topk_indices, random_val,
                           topp,
                           topk_, reinterpret_cast<kunlunStream_t>(stream));
                return INFINI_STATUS_SUCCESS;
            } else {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            break;
        case INFINI_DTYPE_F32:
            CHECK_STATUS(internal->useXdnn(
                (kunlunStream_t)stream,
                [&](xdnnHandle_t handle) {
                    CHECK_KUNLUN((xdnn::sorted_topk<float>(handle, (float *)probs, topk_values, topk_indices, 1, n, topk_, true, true)));
                    float max_value = 0.0f;
                    xpu_memcpy(&max_value, topk_values, sizeof(float), XPUMemcpyKind::XPU_DEVICE_TO_HOST);
                    CHECK_KUNLUN((xdnn::add_scalar<float>(handle, (float *)probs, probs_F32, max_value, -1.0f, n)));
                    CHECK_KUNLUN((xdnn::mul_scalar<float>(handle, probs_F32, probs_F32, 1.0 / temperature, n)));
                    CHECK_KUNLUN((xdnn::softmax<float>(handle, probs_F32, destination, {n}, 0)));
                    CHECK_KUNLUN((xdnn::cumsum<float>(handle, destination, destination, {n}, false, false, 0)));
                    return INFINI_STATUS_SUCCESS;
                }));

            if (dt_i == INFINI_DTYPE_I64) {
                sample_I64(result, destination, topk_indices, random_val,
                           topp,
                           topk_, reinterpret_cast<kunlunStream_t>(stream));
                return INFINI_STATUS_SUCCESS;
            } else if (dt_i == INFINI_DTYPE_I32) {
                sample_I32(result, destination, topk_indices, random_val,
                           topp,
                           topk_, reinterpret_cast<kunlunStream_t>(stream));
                return INFINI_STATUS_SUCCESS;
            } else {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        int64_t *output = (int64_t *)workspace_value;
        switch (dt_p) {
        case INFINI_DTYPE_F32:
            if (dt_i == INFINI_DTYPE_I64) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::argmax<float>(handle, (float *)probs, (int64_t *)result, {n}, 0)));
                        return INFINI_STATUS_SUCCESS;
                    }));
                return INFINI_STATUS_SUCCESS;
            } else if (dt_i == INFINI_DTYPE_I32) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::argmax<float>(handle, (float *)probs, output, {n}, 0)));
                        CHECK_KUNLUN((xdnn::cast<int64_t, int>(handle, output, (int *)result, 1)));
                        return INFINI_STATUS_SUCCESS;
                    }));
                return INFINI_STATUS_SUCCESS;
            } else {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        case INFINI_DTYPE_F16:
            if (dt_i == INFINI_DTYPE_I64) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::argmax<float16>(handle, (float16 *)probs, (int64_t *)result, {n}, 0)));
                        return INFINI_STATUS_SUCCESS;
                    }));
                return INFINI_STATUS_SUCCESS;
            } else if (dt_i == INFINI_DTYPE_I32) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::argmax<float16>(handle, (float16 *)probs, output, {n}, 0)));
                        CHECK_KUNLUN((xdnn::cast<int64_t, int>(handle, output, (int *)result, 1)));
                        return INFINI_STATUS_SUCCESS;
                    }));
                return INFINI_STATUS_SUCCESS;
            } else {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }
}

infiniStatus_t
Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_STATUS(random_sample_kernel(workspace,
                                      workspace_size,
                                      _opaque->internal,
                                      _info.dt_p,
                                      _info.dt_i,
                                      result,
                                      probs,
                                      random_val,
                                      topp,
                                      topk,
                                      temperature,
                                      _info.n,
                                      stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::kunlun
