#include "affine_grid_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../handle.h"

namespace op::affine_grid::nvidia {

// ==================================================================
// Kernel Launch Helper
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input, // 这里对应 Theta
    size_t batch,
    size_t height,
    size_t width,
    bool align_corners,
    void *stream) {

    // 指针强转
    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    // 计算总线程数: N * H * W
    // 每个线程负责生成一个 (x, y) 坐标对
    size_t total_elements = batch * height * width;

    // CUDA Grid/Block 配置
    // 256 是 Memory Bound 算子的推荐 Block Size，能提供较好的 Occupancy
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 调用 cuda 命名空间下的 kernel (定义在 ../cuda/kernel.cuh 中)
    // 注意：kernel 内部会将 T* 强转为 vector type (如 float2*) 以加速写入
    cuda::affine_grid_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr,
        in_ptr,
        batch,
        height,
        width,
        align_corners
    );
}

// ==================================================================
// Descriptor Implementation
// ==================================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

// 创建算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    bool align_corners) { 

    // 1. 使用 Info 类解析并校验参数
    // create 方法内部会检查 Tensor 维度和类型一致性
    auto info_result = AffineGridInfo::create(out_desc, in_desc, align_corners);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),        // Opaque 指针
        info,                // Info 对象 (包含 N, H, W, align_corners)
        0,                   // Workspace size (AffineGrid 不需要额外 workspace)
        handle->device,      // Device Type
        handle->device_id    // Device ID
    );

    return INFINI_STATUS_SUCCESS;
}

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    // 从 _info 中提取必要的维度参数和配置
    auto dtype = _info.dtype();
    auto batch = _info.batch();
    auto height = _info.height();
    auto width = _info.width();
    auto align_corners = _info.align_corners();

    // 根据数据类型分发到对应的模板实例化函数
    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, batch, height, width, align_corners, stream);
        break;
        
    case INFINI_DTYPE_BF16:
        // 使用标准类型 nv_bfloat16 (需要 cuda_bf16.h 支持)
        launch_kernel<nv_bfloat16>(output, input, batch, height, width, align_corners, stream);
        break;
        
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, batch, height, width, align_corners, stream);
        break;
        
    case INFINI_DTYPE_F64:
        // Double 类型通常没有 vector type write (double2 是 128bit)，
        // kernel 会自动回退到 scalar 写入路径，逻辑依然正确。
        launch_kernel<double>(output, input, batch, height, width, align_corners, stream);
        break;
        
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::affine_grid::nvidia