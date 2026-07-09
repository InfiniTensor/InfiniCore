#include "../../../handle.h"
#include "dsv4_sglang_topk_v2_nvidia.cuh"
#if defined(ENABLE_HYGON_API)
#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <tvm/ffi/container/tensor.h>
namespace {
using plan_fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, long);
using transform_fn_t = void (*)(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView, long, tvm::ffi::TensorView, tvm::ffi::TensorView);
constexpr const char *kPlanSymbol = "_Z19sglang_topk_v2_planN3tvm3ffi10TensorViewES1_l";
constexpr const char *kTransformSymbol = "_Z28sglang_topk_v2_transform_512N3tvm3ffi10TensorViewES1_S1_S1_lS1_S1_";
const char *deepseekOpsPath() {
    const char *env = std::getenv("DEEPSEEK_V4_OPS_SO");
    return env && env[0] ? env : "libdeepseek_v4_ops.so";
}
void *tryDlopen(const char *path, int flags) {
    dlerror();
    return dlopen(path, flags);
}
void *handle() {
    static void *h = []() -> void * { tryDlopen("libpython3.10.so.1.0", RTLD_LAZY | RTLD_GLOBAL); tryDlopen("libtorch_python.so", RTLD_LAZY | RTLD_GLOBAL); return tryDlopen(deepseekOpsPath(), RTLD_LAZY | RTLD_GLOBAL); }();
    return h;
}
plan_fn_t resolvePlan() {
    void *h = handle();
    return h ? reinterpret_cast<plan_fn_t>(dlsym(h, kPlanSymbol)) : nullptr;
}
transform_fn_t resolveTransform() {
    void *h = handle();
    return h ? reinterpret_cast<transform_fn_t>(dlsym(h, kTransformSymbol)) : nullptr;
}
template <size_t N>
tvm::ffi::TensorView makeView(const void *data, DLDataType dtype, const std::array<int64_t, N> &shape, const std::array<int64_t, N> &strides, int dev, DLTensor &t) {
    t.data = const_cast<void *>(data);
    t.device = DLDevice{kDLROCM, dev};
    t.ndim = N;
    t.dtype = dtype;
    t.shape = const_cast<int64_t *>(shape.data());
    t.strides = const_cast<int64_t *>(strides.data());
    t.byte_offset = 0;
    return tvm::ffi::TensorView(&t);
}
} // namespace
#endif
namespace op::dsv4_sglang_topk_v2::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t transform_workspace_desc, infiniopTensorDescriptor_t metadata_desc, int64_t page_size) {
    Info info;
    CHECK_STATUS(createInfo(&info, scores_desc, seq_lens_desc, page_table_desc, page_indices_desc, transform_workspace_desc, metadata_desc, page_size));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *transform_workspace, void *metadata, void *) const {
#if defined(ENABLE_HYGON_API)
    auto plan = resolvePlan();
    auto transform = resolveTransform();
    CHECK_OR_RETURN(plan != nullptr && transform != nullptr, INFINI_STATUS_INTERNAL_ERROR);
    const auto &i = _info;
    int dev = this->device_id;
    std::array<int64_t, 2> scores_shape{static_cast<int64_t>(i.batch), 512};
    std::array<int64_t, 2> scores_strides{512, 1};
    std::array<int64_t, 1> seq_shape{static_cast<int64_t>(i.batch)};
    std::array<int64_t, 1> seq_strides{1};
    std::array<int64_t, 2> table_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.pages)};
    std::array<int64_t, 2> table_strides{static_cast<int64_t>(i.pages), 1};
    std::array<int64_t, 2> out_shape{static_cast<int64_t>(i.batch), 512};
    std::array<int64_t, 2> out_strides{512, 1};
    std::array<int64_t, 2> work_shape{static_cast<int64_t>(i.batch), static_cast<int64_t>(i.workspace_width)};
    std::array<int64_t, 2> work_strides{static_cast<int64_t>(i.workspace_width), 1};
    std::array<int64_t, 2> meta_shape{static_cast<int64_t>(i.metadata_rows), 4};
    std::array<int64_t, 2> meta_strides{4, 1};
    DLTensor scores_t, seq_t, table_t, out_t, work_t, meta_t;
    auto scores_v = makeView(scores, DLDataType{kDLFloat, 32, 1}, scores_shape, scores_strides, dev, scores_t);
    auto seq_v = makeView(seq_lens, DLDataType{kDLInt, 32, 1}, seq_shape, seq_strides, dev, seq_t);
    auto table_v = makeView(page_table, DLDataType{kDLInt, 32, 1}, table_shape, table_strides, dev, table_t);
    auto out_v = makeView(page_indices, DLDataType{kDLInt, 32, 1}, out_shape, out_strides, dev, out_t);
    auto work_v = makeView(transform_workspace, DLDataType{kDLInt, 32, 1}, work_shape, work_strides, dev, work_t);
    auto meta_v = makeView(metadata, DLDataType{kDLInt, 32, 1}, meta_shape, meta_strides, dev, meta_t);
    plan(seq_v, meta_v, 0);
    transform(scores_v, seq_v, table_v, out_v, static_cast<long>(i.page_size), work_v, meta_v);
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
}
} // namespace op::dsv4_sglang_topk_v2::nvidia
