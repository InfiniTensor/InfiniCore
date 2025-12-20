#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "rearrange_kernel.cuh"
#include "rearrange_transpose_kernel.cuh"
#include "rearrange_nvidia.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

namespace op::rearrange::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();

    CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);
    // 保存临时vector对象
    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    auto y_strides = y_desc->strides();
    auto x_strides = x_desc->strides();

    CHECK_SAME_SHAPE(x_shape, y_shape);

    auto meta = utils::RearrangeMeta::create(
        y_shape.data(),
        y_strides.data(),
        x_strides.data(),
        ndim,
        infiniSizeOf(dtype));

    CHECK_RESULT(meta);

    *desc_ptr = new Descriptor(
        std::move(*meta),
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 维度信息结构
struct Dim {
    size_t len;
    ARRAY_TYPE_STRIDE src_stride;
    ARRAY_TYPE_STRIDE dst_stride;
};

// 分割维度结构
struct SplitDim {
    size_t choose_idx;
    size_t num_per_block;
    size_t num_per_grid;
    int array_struct_idx_block;
    int array_struct_idx_grid;
    size_t dim_len;
};

/**
 * 检测是否为完全转置模式（行主序到列主序，或反之）
 * 
 * 判断逻辑：
 * 1. 检查src_strides和dst_strides是否呈现相反的递增/递减趋势
 * 2. 对于行主序到列主序转换：src_strides递减，dst_strides递增
 * 3. 只对满足条件的大规模转置启用优化
 * 
 * @param meta 重排元数据
 * @return true如果是完全转置模式且适合优化
 */
bool isFullTransposePattern(const utils::RearrangeMeta &meta) {
    const size_t ndim = meta.ndim();
    
    // 只针对特定维度范围启用转置优化
    // 避免对小规模或不适合的case使用
    if (ndim < 4 || ndim > 6) {
        return false;
    }
    
    const ptrdiff_t *src_strides = meta.src_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const size_t unit = meta.unit();
    
    // 构建实际的shape
    std::vector<size_t> shape(ndim);
    auto prev_idx_stride = meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = prev_idx_stride / idx_strides[i];
        prev_idx_stride = idx_strides[i];
    }
    
    // 计算总元素数，只对大规模转置启用优化
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= shape[i];
    }
    
    // 只对足够规模的数据启用转置优化
    // 6D: 100K+; 5D: 20K+（用于覆盖 (3,4,7,53,9) 这类中等规模 full-transpose）; 4D: 50K+
    size_t threshold = 100000;
    if (ndim == 5) threshold = 20000;
    if (ndim == 4) threshold = 50000;
    if (total_elements < threshold) return false;
    
    // 跳过大小为1的维度，构建有效维度的索引
    std::vector<size_t> valid_dims;
    for (size_t i = 0; i < ndim; ++i) {
        if (shape[i] > 1) {
            valid_dims.push_back(i);
        }
    }
    
    // 至少需要4个有效维度才启用转置优化
    if (valid_dims.size() < 4) {
        return false;
    }
    
    // 检查是否为row-major到column-major的完全转换
    // 这是最容易识别的模式：src_strides递减，dst_strides递增（或相反）
    
    // 计算stride的排序
    std::vector<std::pair<ptrdiff_t, size_t>> src_stride_order;
    std::vector<std::pair<ptrdiff_t, size_t>> dst_stride_order;
    
    for (size_t i = 0; i < valid_dims.size(); ++i) {
        size_t dim = valid_dims[i];
        src_stride_order.push_back({std::abs(src_strides[dim]), dim});
        dst_stride_order.push_back({std::abs(dst_strides[dim]), dim});
    }
    
    std::sort(src_stride_order.begin(), src_stride_order.end());
    std::sort(dst_stride_order.begin(), dst_stride_order.end());
    
    // 检查排序后的维度顺序是否完全相反
    bool is_reversed = true;
    for (size_t i = 0; i < valid_dims.size(); ++i) {
        if (src_stride_order[i].second != dst_stride_order[valid_dims.size() - 1 - i].second) {
            is_reversed = false;
            break;
        }
    }
    
    return is_reversed;
}

/**
 * 根据给定的元数据准备张量重排参数，该函数主要完成以下工作：
 * 1. 根据原始元数据调整单元大小，获取更适合GPU处理的单元大小
 * 2. 将维度分配为CUDA块（block）维度和网格（grid）维度：
 *    该步骤是核心，目标是为每个block分配尽可能多的相对连续的数据进行处理，
 *    对无法完整放入块的维度进行分割，并记录分割维度信息，用于防止kernel访问越界，最大化内存访问局部性和计算效率
 */
utils::Result<RearrangeParams> prepareRearrangeParams(const utils::RearrangeMeta &original_meta, int max_threads) {
    RearrangeParams params;

    // 获取更适合GPU处理的单元大小，这里使用2的幂次方
    auto meta_result = original_meta.distributeUnit({32, 16, 8, 4, 2, 1});
    CHECK_RESULT(meta_result);
    const utils::RearrangeMeta &meta = meta_result.take();

    // 获取维度信息
    const size_t ndim = meta.ndim();
    const size_t unit = meta.unit();

    // 特殊情况：无维度，只需要简单复制
    if (ndim == 0) {
        params.block_dim = 0;
        params.block_len_total = 1;
        params.block_len = {static_cast<ARRAY_TYPE_SIZE>(1)};
        params.src_block_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.dst_block_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.grid_len = {static_cast<ARRAY_TYPE_SIZE>(1)};
        params.src_grid_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.dst_grid_stride = {static_cast<ARRAY_TYPE_STRIDE>(0)};
        params.unit_size = unit;
        return utils::Result<RearrangeParams>(params);
    }

    // 从元数据中提取必要的信息
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *src_strides = meta.src_strides();

    // 准备维度信息
    std::vector<Dim> dims;
    std::vector<size_t> shape;
    dims.reserve(ndim);
    shape.reserve(ndim);

    auto prev_idx_stride = meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        size_t len = prev_idx_stride / idx_strides[i];
        shape.push_back(len);
        dims.push_back({len, src_strides[i], dst_strides[i]});
        prev_idx_stride = idx_strides[i];
    }

    std::vector<bool> block_dim_choose(ndim, false);
    std::vector<SplitDim> split_dims;

    // 初始化计数器
    size_t block_elements = 1;

    std::vector<size_t> dim_order(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        dim_order[i] = i;
    }

    // 按src_stride升序排序，贪心选择
    std::sort(dim_order.begin(), dim_order.end(),
              [&dims](size_t a, size_t b) {
                  return std::abs(dims[a].src_stride) < std::abs(dims[b].src_stride);
              });

    // 辅助函数：检查block_size是否是warp-friendly的
    auto is_warp_friendly = [](size_t size) -> bool {
        // 检查是否是32的倍数，或者是常见的高效配置
        if (size % 32 == 0) return true;
        // 小于32的2的幂次也是可以的
        if (size <= 32 && (size & (size - 1)) == 0) return true;
        return false;
    };

    // 辅助函数：计算warp效率损失
    auto warp_efficiency = [](size_t size) -> double {
        if (size == 0) return 0.0;
        size_t warps = (size + 31) / 32;
        size_t wasted = warps * 32 - size;
        return 1.0 - (double)wasted / (double)(warps * 32);
    };

    // 维度选择循环 - 带warp对齐优化
    for (size_t i = 0; i < ndim; ++i) {
        size_t dim_idx = dim_order[i];
        size_t dim_len = shape[dim_idx];
        size_t next_block_size = block_elements * dim_len;

        if (next_block_size <= (size_t)max_threads) {
            // 检查是否应该添加这个维度
            bool should_add = true;

            // 优化策略1: 如果当前已经是warp-friendly且下一个会破坏对齐，考虑跳过
            if (is_warp_friendly(block_elements) && !is_warp_friendly(next_block_size)) {
                // 检查是否接近max_threads且效率会显著下降
                if (next_block_size > max_threads * 0.95) {
                    double current_eff = warp_efficiency(block_elements);
                    double next_eff = warp_efficiency(next_block_size);
                    
                    // 如果效率损失超过5%，不添加这个维度
                    if (current_eff - next_eff > 0.05) {
                        should_add = false;
                    }
                }
            }

            // 优化策略2: 对于会产生接近1024但不对齐的情况，尝试调整
            if (should_add && next_block_size > 992 && next_block_size < 1024 && !is_warp_friendly(next_block_size)) {
                // 如果当前block_elements是warp-friendly的，并且已经足够大（>512），停止添加
                if (is_warp_friendly(block_elements) && block_elements >= 512) {
                    should_add = false;
                }
            }

            if (should_add) {
                block_dim_choose[dim_idx] = true;
                block_elements = next_block_size;
            }
        } else if (block_elements > 1 && dim_len > 1) {
            // 需要分割此维度
            size_t num_per_block = std::min(dim_len, (size_t)max_threads / block_elements);
            
            // 优化分割：尽量让分割后的block_size是32的倍数
            if (num_per_block > 1 && block_elements > 1) {
                size_t target_block_size = block_elements * num_per_block;
                
                // 如果不是warp-friendly，尝试调整num_per_block
                if (!is_warp_friendly(target_block_size) && target_block_size > 32) {
                    // 尝试找到最近的能产生warp-aligned结果的num_per_block
                    for (size_t try_num = num_per_block; try_num > 0; try_num--) {
                        size_t try_size = block_elements * try_num;
                        if (is_warp_friendly(try_size) && try_size >= 512) {
                            num_per_block = try_num;
                            break;
                        }
                    }
                }
            }
            
            if (num_per_block > 0) {
                size_t num_per_grid = (dim_len + num_per_block - 1) / num_per_block;

                SplitDim split_dim = {
                    dim_idx,       // choose_idx
                    num_per_block, // num_per_block
                    num_per_grid,  // num_per_grid
                    0,             // array_struct_idx_block (待更新)
                    0,             // array_struct_idx_grid (待更新)
                    dim_len        // original dimension length
                };
                split_dims.push_back(split_dim);
                block_elements *= num_per_block;
            }
            break;
        }
    }

    if (block_elements == 1 && ndim > 0) {
        size_t dim_idx = dim_order[0];
        size_t dim_len = shape[dim_idx];

        if (dim_len <= (size_t)max_threads) {
            // 优化：如果dim_len不是warp-friendly，尝试调整到最近的warp边界
            if (!is_warp_friendly(dim_len) && dim_len > 32) {
                // 尝试选择32的倍数
                size_t aligned_len = (dim_len / 32) * 32;
                if (aligned_len >= 512 && aligned_len <= (size_t)max_threads) {
                    // 使用分割策略
                    size_t num_per_grid = (dim_len + aligned_len - 1) / aligned_len;
                    SplitDim split_dim = {
                        dim_idx,
                        aligned_len,
                        num_per_grid,
                        0,
                        0,
                        dim_len};
                    split_dims.push_back(split_dim);
                    block_elements = aligned_len;
                } else {
                    block_dim_choose[dim_idx] = true;
                    block_elements = dim_len;
                }
            } else {
                block_dim_choose[dim_idx] = true;
                block_elements = dim_len;
            }
        } else {
            // 需要分割
            size_t num_per_block = std::min(dim_len, (size_t)max_threads);
            
            // 优化：优先选择32的倍数
            if (!is_warp_friendly(num_per_block) && num_per_block > 32) {
                size_t aligned = (num_per_block / 32) * 32;
                if (aligned >= 512) {
                    num_per_block = aligned;
                }
            }
            
            size_t num_per_grid = (dim_len + num_per_block - 1) / num_per_block;

            SplitDim split_dim = {
                dim_idx,
                num_per_block,
                num_per_grid,
                0,
                0,
                dim_len};
            split_dims.push_back(split_dim);
            block_elements = num_per_block;
        }
    }

    // 准备block维度相关参数
    size_t block_dim = 0;
    size_t block_len_total = block_elements;

    std::vector<ARRAY_TYPE_SIZE> block_len;
    std::vector<ARRAY_TYPE_STRIDE> src_block_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_block_stride;

    std::vector<ARRAY_TYPE_SIZE> grid_len;
    std::vector<ARRAY_TYPE_STRIDE> src_grid_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_grid_stride;

    // 处理block维度，填充block_len和block_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (block_dim_choose[i]) {
            block_len.push_back(shape[i]);
            src_block_stride.push_back(dims[i].src_stride);
            dst_block_stride.push_back(dims[i].dst_stride);
            block_dim += 1;
        }

        // 处理分割维度的block部分
        for (size_t j = 0; j < split_dims.size(); ++j) {
            if (i == split_dims[j].choose_idx) {
                block_len.push_back(split_dims[j].num_per_block);
                src_block_stride.push_back(dims[i].src_stride);
                dst_block_stride.push_back(dims[i].dst_stride);
                split_dims[j].array_struct_idx_block = static_cast<int>(block_dim);
                block_dim += 1;
            }
        }
    }

    // 处理grid维度，填充grid_len和grid_stride
    for (size_t i = 0; i < ndim; ++i) {
        if (!block_dim_choose[i]) {
            bool is_split = false;

            // 检查是否是分割维度
            for (size_t j = 0; j < split_dims.size(); ++j) {
                if (i == split_dims[j].choose_idx) {
                    is_split = true;
                    grid_len.push_back(split_dims[j].num_per_grid);
                    src_grid_stride.push_back(dims[i].src_stride * split_dims[j].num_per_block);
                    dst_grid_stride.push_back(dims[i].dst_stride * split_dims[j].num_per_block);
                    split_dims[j].array_struct_idx_grid = static_cast<int>(grid_len.size() - 1);
                    break;
                }
            }

            // 如果不是分割维度，则作为完整的grid维度
            if (!is_split) {
                grid_len.push_back(shape[i]);
                src_grid_stride.push_back(dims[i].src_stride);
                dst_grid_stride.push_back(dims[i].dst_stride);
            }
        }
    }

    // 如果grid_len为空，添加一个默认值
    if (grid_len.empty()) {
        grid_len.push_back(1);
        src_grid_stride.push_back(0);
        dst_grid_stride.push_back(0);
    }

    // 处理约束条件 - 使用与Rust版本相似的逻辑
    std::vector<Constraint<ARRAY_TYPE_SIZE>> constraints;

    // 限制最多处理2个约束条件
    for (size_t i = 0; i < split_dims.size(); ++i) {
        if (split_dims[i].dim_len % split_dims[i].num_per_block == 0) {
            continue;
        }
        Constraint<ARRAY_TYPE_SIZE> constraint;
        constraint.grid_idx = split_dims[i].array_struct_idx_grid;
        constraint.block_idx = split_dims[i].array_struct_idx_block;
        constraint.grid_div_block = split_dims[i].num_per_block;
        constraint.total_len = split_dims[i].dim_len;
        constraints.push_back(constraint);

        if (constraints.size() >= 2) {
            break;
        }
    }

    // 设置参数
    params.block_dim = block_dim;
    params.block_len_total = block_len_total;
    params.block_len = block_len;
    params.src_block_stride = src_block_stride;
    params.dst_block_stride = dst_block_stride;
    params.grid_len = grid_len;
    params.src_grid_stride = src_grid_stride;
    params.dst_grid_stride = dst_grid_stride;
    params.constraints = constraints;
    params.unit_size = unit;

    return utils::Result<RearrangeParams>(params);
}

// ==============================================================================
// 动态Kernel启动函数 - 支持任意维度
// ==============================================================================

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchDynamicKernel(
    void *y,
    const void *x,
    size_t grid_size,
    const RearrangeParams &params,
    size_t unit_size,
    cudaStream_t stream) {

    // 检查参数有效性
    if (params.block_len.empty() || params.grid_len.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    size_t block_dim = params.block_len.size();
    size_t grid_dim = params.grid_len.size();
    size_t block_len_total = params.block_len_total;
    size_t constraint_num = params.constraints.size();

    // 准备device端数组
    ARRAY_TYPE_SIZE *d_block_len = nullptr;
    ARRAY_TYPE_STRIDE *d_src_block_stride = nullptr;
    ARRAY_TYPE_STRIDE *d_dst_block_stride = nullptr;
    ARRAY_TYPE_SIZE *d_grid_len = nullptr;
    ARRAY_TYPE_STRIDE *d_src_grid_stride = nullptr;
    ARRAY_TYPE_STRIDE *d_dst_grid_stride = nullptr;
    Constraint<ARRAY_TYPE_SIZE> *d_constraints = nullptr;

    // 分配设备内存
    CHECK_OR_RETURN(cudaMalloc(&d_block_len, block_dim * sizeof(ARRAY_TYPE_SIZE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMalloc(&d_src_block_stride, block_dim * sizeof(ARRAY_TYPE_STRIDE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMalloc(&d_dst_block_stride, block_dim * sizeof(ARRAY_TYPE_STRIDE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMalloc(&d_grid_len, grid_dim * sizeof(ARRAY_TYPE_SIZE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMalloc(&d_src_grid_stride, grid_dim * sizeof(ARRAY_TYPE_STRIDE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMalloc(&d_dst_grid_stride, grid_dim * sizeof(ARRAY_TYPE_STRIDE)) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);

    // 使用异步拷贝提升性能
    CHECK_OR_RETURN(cudaMemcpyAsync(d_block_len, params.block_len.data(), 
                                     block_dim * sizeof(ARRAY_TYPE_SIZE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMemcpyAsync(d_src_block_stride, params.src_block_stride.data(), 
                                     block_dim * sizeof(ARRAY_TYPE_STRIDE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMemcpyAsync(d_dst_block_stride, params.dst_block_stride.data(), 
                                     block_dim * sizeof(ARRAY_TYPE_STRIDE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMemcpyAsync(d_grid_len, params.grid_len.data(), 
                                     grid_dim * sizeof(ARRAY_TYPE_SIZE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMemcpyAsync(d_src_grid_stride, params.src_grid_stride.data(), 
                                     grid_dim * sizeof(ARRAY_TYPE_STRIDE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);
    CHECK_OR_RETURN(cudaMemcpyAsync(d_dst_grid_stride, params.dst_grid_stride.data(), 
                                     grid_dim * sizeof(ARRAY_TYPE_STRIDE), 
                                     cudaMemcpyHostToDevice, stream) == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);

    // 处理约束
    if (constraint_num > 0) {
        CHECK_OR_RETURN(cudaMalloc(&d_constraints, constraint_num * sizeof(Constraint<ARRAY_TYPE_SIZE>)) == cudaSuccess,
                        INFINI_STATUS_INTERNAL_ERROR);
        CHECK_OR_RETURN(cudaMemcpyAsync(d_constraints, params.constraints.data(), 
                                         constraint_num * sizeof(Constraint<ARRAY_TYPE_SIZE>), 
                                         cudaMemcpyHostToDevice, stream) == cudaSuccess,
                        INFINI_STATUS_INTERNAL_ERROR);
    }

    // 根据unit_size选择合适的kernel
    void *kernel_func = nullptr;
    switch (unit_size) {
    case 1:
        kernel_func = (void *)rearrange_dynamic_kernel<uchar1>;
        break;
    case 2:
        kernel_func = (void *)rearrange_dynamic_kernel<uchar2>;
        break;
    case 4:
        kernel_func = (void *)rearrange_dynamic_kernel<float1>;
        break;
    case 8:
        kernel_func = (void *)rearrange_dynamic_kernel<float2>;
        break;
    case 16:
        kernel_func = (void *)rearrange_dynamic_kernel<float4>;
        break;
    case 32:
        kernel_func = (void *)rearrange_dynamic_kernel<double4_32a>;
        break;
    default:
        return INFINI_STATUS_BAD_PARAM;
    }

    // 准备kernel参数
    void *args[] = {
        &y, &x,
        const_cast<size_t *>(&block_dim),
        const_cast<size_t *>(&block_len_total),
        &d_block_len,
        &d_src_block_stride,
        &d_dst_block_stride,
        const_cast<size_t *>(&grid_dim),
        &d_grid_len,
        &d_src_grid_stride,
        &d_dst_grid_stride,
        const_cast<size_t *>(&constraint_num),
        &d_constraints};

    // 启动kernel
    cudaError_t launch_result = cudaLaunchKernel(
        kernel_func,
        static_cast<unsigned int>(grid_size),
        static_cast<unsigned int>(BLOCK_SIZE),
        args, 0, stream);

    // 检查kernel启动是否成功
    if (launch_result != cudaSuccess) {
        // 清理设备内存
        cudaFree(d_block_len);
        cudaFree(d_src_block_stride);
        cudaFree(d_dst_block_stride);
        cudaFree(d_grid_len);
        cudaFree(d_src_grid_stride);
        cudaFree(d_dst_grid_stride);
        if (d_constraints) {
            cudaFree(d_constraints);
        }
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    // 同步stream确保kernel完成后再释放内存
    // 注意：cudaFree会隐式同步，所以这里不需要显式cudaStreamSynchronize
    
    // 清理设备内存
    cudaFree(d_block_len);
    cudaFree(d_src_block_stride);
    cudaFree(d_dst_block_stride);
    cudaFree(d_grid_len);
    cudaFree(d_src_grid_stride);
    cudaFree(d_dst_grid_stride);
    if (d_constraints) {
        cudaFree(d_constraints);
    }

    return INFINI_STATUS_SUCCESS;
}

// ==============================================================================
// 静态Kernel启动函数 - 为常见维度组合优化
// ==============================================================================

// 带约束的内核启动模板函数
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    void *y,
    const void *x,
    size_t grid_size,
    const RearrangeParams &params,
    size_t unit_size,
    cudaStream_t stream) {

    // 获取内核函数
    RearrangeParams params_copy = params; // 创建一个非const副本
    auto kernel_func_result = getRearrangeKernel(params_copy);

    CHECK_RESULT(kernel_func_result);
    auto kernel_func = kernel_func_result.take();

    // 创建非const的临时变量
    size_t block_dim = params.block_dim;
    size_t block_len_total = params.block_len_total;

    // 检查向量尺寸是否合理
    if (params.block_len.size() < block_dim || params.src_block_stride.size() < block_dim || params.dst_block_stride.size() < block_dim) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (params.grid_len.empty() || params.src_grid_stride.empty() || params.dst_grid_stride.empty()) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const Constraint<ARRAY_TYPE_SIZE> *constraints_data;
    auto empty_constraints = Constraint<ARRAY_TYPE_SIZE>();
    if (params.constraints.empty()) {
        constraints_data = &empty_constraints;
    } else {
        constraints_data = params.constraints.data();
    }

    void *args[]
        = {
            &y, &x,
            &block_dim,
            &block_len_total,
            const_cast<void *>(static_cast<const void *>(params.block_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_block_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.grid_len.data())),
            const_cast<void *>(static_cast<const void *>(params.src_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(params.dst_grid_stride.data())),
            const_cast<void *>(static_cast<const void *>(constraints_data))};

    CHECK_OR_RETURN(cudaLaunchKernel(
                        kernel_func,
                        static_cast<unsigned int>(grid_size), static_cast<unsigned int>(BLOCK_SIZE),
                        args, 0, stream)
                        == cudaSuccess,
                    INFINI_STATUS_INTERNAL_ERROR);

    return INFINI_STATUS_SUCCESS;
}

/**
 * 启动转置优化的kernel
 * 针对完全转置场景使用优化的实现
 */
infiniStatus_t launchTransposeKernel(
    void *y,
    const void *x,
    const utils::RearrangeMeta &meta,
    cudaStream_t stream) {
    
    const size_t ndim = meta.ndim();
    const size_t unit = meta.unit();
    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *src_strides = meta.src_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    
    // 构建shape
    std::vector<size_t> shape(ndim);
    auto prev_idx_stride = meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = prev_idx_stride / idx_strides[i];
        prev_idx_stride = idx_strides[i];
    }
    
    // 计算总元素数
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_elements *= shape[i];
    }

    // 2D 大矩阵 row<->col major layout transform：
    // - src column-major (stride=(1, M)) -> dst row-major (stride=(N, 1))
    // - src row-major (stride=(N, 1)) -> dst column-major (stride=(1, M))
    // 这类 case 本质等价于一次 2D transpose，必须用 tiled transpose 才能接近带宽上限
    // 2D transpose-like layout transforms:
    // Only enable for sufficiently large matrices; for small sizes (e.g., 100x100),
    // the generic rearrange kernel can be faster than a shared-memory tiled transpose.
    if (ndim == 2 && total_elements >= 65536 && (unit == 2 || unit == 4)) {
        const size_t d0 = shape[0]; // M
        const size_t d1 = shape[1]; // N

        const ptrdiff_t s0 = src_strides[0];
        const ptrdiff_t s1 = src_strides[1];
        const ptrdiff_t t0 = dst_strides[0];
        const ptrdiff_t t1 = dst_strides[1];

        // 只支持正 stride（负 stride 暂不处理）
        if (s0 > 0 && s1 > 0 && t0 > 0 && t1 > 0) {
            const ptrdiff_t u = static_cast<ptrdiff_t>(unit);

            // Pattern A: src col-major -> dst row-major
            // src: (1, d0) * unit, dst: (d1, 1) * unit
            const bool src_is_col_major = (s0 == u) && (s1 == static_cast<ptrdiff_t>(d0) * u);
            const bool dst_is_row_major = (t0 == static_cast<ptrdiff_t>(d1) * u) && (t1 == u);

            // Pattern B: src row-major -> dst col-major
            // src: (d1, 1) * unit, dst: (1, d0) * unit
            const bool src_is_row_major = (s0 == static_cast<ptrdiff_t>(d1) * u) && (s1 == u);
            const bool dst_is_col_major = (t0 == u) && (t1 == static_cast<ptrdiff_t>(d0) * u);

            // Optional one-shot debug for pattern matching
            static bool transpose_debug_enabled = []() {
                const char* env = std::getenv("REARRANGE_DEBUG_TRANSPOSE");
                return env != nullptr && std::string(env) == "1";
            }();
            static bool transpose_debug_printed = false;
            if (transpose_debug_enabled && !transpose_debug_printed) {
                transpose_debug_printed = true;
                printf("\n=== Rearrange Transpose Debug ===\n");
                printf("ndim=2, unit=%zu, shape=(%zu,%zu), total=%zu\n", unit, d0, d1, total_elements);
                printf("src_strides(bytes)=(%td,%td), dst_strides(bytes)=(%td,%td)\n", s0, s1, t0, t1);
                printf("match: src_col=%d dst_row=%d src_row=%d dst_col=%d\n",
                       (int)src_is_col_major, (int)dst_is_row_major, (int)src_is_row_major, (int)dst_is_col_major);
                printf("===============================\n");
            }

            // block=(32,8) + shared-memory tile，来自 CUDA transpose sample
            dim3 block(TILE_DIM, BLOCK_ROWS, 1);
            dim3 block_small(TILE_DIM_SMALL, BLOCK_ROWS_SMALL, 1);

            if (src_is_col_major && dst_is_row_major) {
                // 解释为：transpose A(N,M)->B(M,N) 的 contiguous row-major transpose
                const size_t rows = d1; // N
                const size_t cols = d0; // M
                const bool use_small = (rows <= 256 && cols <= 256);
                dim3 grid(
                    (cols + (use_small ? TILE_DIM_SMALL : TILE_DIM) - 1) / (use_small ? TILE_DIM_SMALL : TILE_DIM),
                    (rows + (use_small ? TILE_DIM_SMALL : TILE_DIM) - 1) / (use_small ? TILE_DIM_SMALL : TILE_DIM),
                    1);

                const ptrdiff_t src_row = src_strides[1] / unit; // M
                const ptrdiff_t src_col = src_strides[0] / unit; // 1
                const ptrdiff_t dst_row = dst_strides[0] / unit; // N
                const ptrdiff_t dst_col = dst_strides[1] / unit; // 1

                if (unit == 4) {
                    using T = uint32_t;
                    if (use_small) {
                        transpose_2d_kernel_tiled_small<T><<<grid, block_small, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    } else {
                        transpose_2d_kernel_tiled<T><<<grid, block, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    }
                } else {
                    using T = uint16_t;
                    if (use_small) {
                        transpose_2d_kernel_tiled_small<T><<<grid, block_small, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    } else {
                        transpose_2d_kernel_tiled<T><<<grid, block, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    }
                }
                CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, INFINI_STATUS_INTERNAL_ERROR);
                return INFINI_STATUS_SUCCESS;
            }

            if (src_is_row_major && dst_is_col_major) {
                // 解释为：transpose A(M,N)->B(N,M)，写入到 dst 的 col-major 布局（等价 row-major N×M）
                const size_t rows = d0; // M
                const size_t cols = d1; // N
                const bool use_small = (rows <= 256 && cols <= 256);
                dim3 grid(
                    (cols + (use_small ? TILE_DIM_SMALL : TILE_DIM) - 1) / (use_small ? TILE_DIM_SMALL : TILE_DIM),
                    (rows + (use_small ? TILE_DIM_SMALL : TILE_DIM) - 1) / (use_small ? TILE_DIM_SMALL : TILE_DIM),
                    1);

                const ptrdiff_t src_row = src_strides[0] / unit; // N
                const ptrdiff_t src_col = src_strides[1] / unit; // 1
                const ptrdiff_t dst_row = dst_strides[1] / unit; // M
                const ptrdiff_t dst_col = dst_strides[0] / unit; // 1

                if (unit == 4) {
                    using T = uint32_t;
                    if (use_small) {
                        transpose_2d_kernel_tiled_small<T><<<grid, block_small, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    } else {
                        transpose_2d_kernel_tiled<T><<<grid, block, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    }
                } else {
                    using T = uint16_t;
                    if (use_small) {
                        transpose_2d_kernel_tiled_small<T><<<grid, block_small, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    } else {
                        transpose_2d_kernel_tiled<T><<<grid, block, 0, stream>>>(
                            reinterpret_cast<T *>(y),
                            reinterpret_cast<const T *>(x),
                            rows, cols,
                            src_row, src_col,
                            dst_row, dst_col);
                    }
                }
                CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, INFINI_STATUS_INTERNAL_ERROR);
                return INFINI_STATUS_SUCCESS;
            }
        }
    }

    // 根据ndim和unit选择合适的kernel
    if (ndim == 5 && total_elements > 20000 && (unit == 2 || unit == 4)) {
        constexpr int VEC = 4;
        const int threads = 256;
        const int blocks = (static_cast<int>((total_elements + VEC - 1) / VEC) + threads - 1) / threads;

        if (unit == 4) {
            auto *src_f32 = reinterpret_cast<float *>(const_cast<void *>(x));
            auto *dst_f32 = reinterpret_cast<float *>(y);

            transpose_5d_kernel_inc<float, VEC><<<blocks, threads, 0, stream>>>(
                dst_f32, src_f32,
                shape[0], shape[1], shape[2], shape[3], shape[4],
                src_strides[0] / unit, src_strides[1] / unit, src_strides[2] / unit, src_strides[3] / unit, src_strides[4] / unit,
                dst_strides[0] / unit, dst_strides[1] / unit, dst_strides[2] / unit, dst_strides[3] / unit, dst_strides[4] / unit,
                total_elements);
        } else {
            using T = uint16_t;
            auto *src_u16 = reinterpret_cast<T *>(const_cast<void *>(x));
            auto *dst_u16 = reinterpret_cast<T *>(y);

            transpose_5d_kernel_inc<T, VEC><<<blocks, threads, 0, stream>>>(
                dst_u16, src_u16,
                shape[0], shape[1], shape[2], shape[3], shape[4],
                src_strides[0] / unit, src_strides[1] / unit, src_strides[2] / unit, src_strides[3] / unit, src_strides[4] / unit,
                dst_strides[0] / unit, dst_strides[1] / unit, dst_strides[2] / unit, dst_strides[3] / unit, dst_strides[4] / unit,
                total_elements);
        }

        CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, INFINI_STATUS_INTERNAL_ERROR);
        return INFINI_STATUS_SUCCESS;
    }

    if (ndim == 6 && total_elements > 100000 && (unit == 2 || unit == 4)) {
        // 大规模6D转置 - F16/F32使用特化kernel
        constexpr int VEC = 4;
        const int threads = 256;
        const int blocks = (static_cast<int>((total_elements + VEC - 1) / VEC) + threads - 1) / threads;
        
        if (unit == 4) {
            auto *src_f32 = reinterpret_cast<float *>(const_cast<void *>(x));
            auto *dst_f32 = reinterpret_cast<float *>(y);

            transpose_6d_kernel_inc<float, VEC><<<blocks, threads, 0, stream>>>(
                dst_f32, src_f32,
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                src_strides[0] / unit, src_strides[1] / unit, src_strides[2] / unit,
                src_strides[3] / unit, src_strides[4] / unit, src_strides[5] / unit,
                dst_strides[0] / unit, dst_strides[1] / unit, dst_strides[2] / unit,
                dst_strides[3] / unit, dst_strides[4] / unit, dst_strides[5] / unit,
                total_elements);
        } else {
            // unit == 2 : use uint16_t for bitwise copy (float16/bfloat16 are both 2 bytes here)
            using T = uint16_t;
            auto *src_u16 = reinterpret_cast<T *>(const_cast<void *>(x));
            auto *dst_u16 = reinterpret_cast<T *>(y);

            transpose_6d_kernel_inc<T, VEC><<<blocks, threads, 0, stream>>>(
                dst_u16, src_u16,
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                src_strides[0] / unit, src_strides[1] / unit, src_strides[2] / unit,
                src_strides[3] / unit, src_strides[4] / unit, src_strides[5] / unit,
                dst_strides[0] / unit, dst_strides[1] / unit, dst_strides[2] / unit,
                dst_strides[3] / unit, dst_strides[4] / unit, dst_strides[5] / unit,
                total_elements);
        }
            
        CHECK_OR_RETURN(cudaGetLastError() == cudaSuccess, INFINI_STATUS_INTERNAL_ERROR);
        return INFINI_STATUS_SUCCESS;
    }
    
    // 对于其他情况，暂不使用通用转置（性能不够好）
    // 返回错误让它回退到原有实现
    return INFINI_STATUS_BAD_PARAM;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 如果没有维度，直接进行内存拷贝
    if (_meta.ndim() == 0) {
        CHECK_OR_RETURN(cudaMemcpyAsync(y, x, _meta.unit(), cudaMemcpyDeviceToDevice, cuda_stream) == cudaSuccess,
                        INFINI_STATUS_INTERNAL_ERROR);
        return INFINI_STATUS_SUCCESS;
    }

    // 检测是否为完全转置模式：
    // - 2D：row<->col major fast-path
    // - 4D~6D：full-transpose (stride-order reversed) fast-path
    if (_meta.ndim() == 2 || ((_meta.ndim() >= 4 && _meta.ndim() <= 6) && isFullTransposePattern(_meta))) {
        // 使用优化的转置kernel
        auto status = launchTransposeKernel(y, x, _meta, cuda_stream);
        // 如果转置kernel成功，直接返回
        if (status == INFINI_STATUS_SUCCESS) {
            return status;
        }
        // 否则回退到通用实现
    }

    // 获取设备属性
    int max_threads = _opaque->internal->maxThreadsPerBlock();

    // 准备参数
    auto params_result = prepareRearrangeParams(_meta, std::min(CUDA_BLOCK_SIZE_1024, max_threads));
    CHECK_RESULT(params_result);
    auto params = params_result.take();

    // 计算grid大小
    size_t grid_size = 1;
    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        grid_size *= params.grid_len[i];
    }

    // 检查grid大小是否为0
    if (grid_size == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    size_t block_size = params.block_len_total;
    size_t block_dim = params.block_len.size();
    size_t grid_dim = params.grid_len.size();

    // 调试输出（通过环境变量REARRANGE_DEBUG=1启用）
    static bool debug_enabled = []() {
        const char* env = std::getenv("REARRANGE_DEBUG");
        return env != nullptr && std::string(env) == "1";
    }();
    
    if (debug_enabled) {
        printf("\n=== Rearrange Debug Info ===\n");
        printf("ndim: %zu, unit_size: %zu\n", _meta.ndim(), _meta.unit());
        printf("block_dim: %zu, block_size: %zu\n", block_dim, block_size);
        printf("grid_dim: %zu, grid_size: %zu\n", grid_dim, grid_size);
        printf("block_len: [");
        for (size_t i = 0; i < params.block_len.size(); ++i) {
            printf("%zu%s", params.block_len[i], i + 1 < params.block_len.size() ? ", " : "");
        }
        printf("]\n");
        printf("grid_len: [");
        for (size_t i = 0; i < params.grid_len.size(); ++i) {
            printf("%zu%s", params.grid_len[i], i + 1 < params.grid_len.size() ? ", " : "");
        }
        printf("]\n");
        printf("constraints: %zu\n", params.constraints.size());
        printf("============================\n");
    }

    // 检查是否需要使用动态kernel (fallback策略)
    bool use_dynamic_kernel = false;
    
    // 情况1: 维度超出静态kernel的支持范围
    if (block_dim > MAX_BLOCK_ARRAY_SIZE || grid_dim > MAX_GRID_ARRAY_SIZE) {
        use_dynamic_kernel = true;
    }
    
    // 情况2: 约束数量超出静态kernel的支持范围
    if (params.constraints.size() > 2) {
        use_dynamic_kernel = true;
    }

    if (debug_enabled) {
        printf("kernel_type: %s\n", use_dynamic_kernel ? "DYNAMIC" : "STATIC");
        printf("block_size_choice: %s\n", block_size <= CUDA_BLOCK_SIZE_512 ? "512" : "1024");
    }

    infiniStatus_t status = INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    if (use_dynamic_kernel) {
        // 使用动态kernel处理高维度或特殊情况
        if (block_size <= CUDA_BLOCK_SIZE_512) {
            status = launchDynamicKernel<CUDA_BLOCK_SIZE_512>(y, x, grid_size, params, _meta.unit(), cuda_stream);
        } else if (block_size <= CUDA_BLOCK_SIZE_1024) {
            status = launchDynamicKernel<CUDA_BLOCK_SIZE_1024>(y, x, grid_size, params, _meta.unit(), cuda_stream);
        } else {
            return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
        }
    } else {
        // 使用静态优化kernel处理常见情况
        if (block_size <= CUDA_BLOCK_SIZE_512) {
            status = launchKernel<CUDA_BLOCK_SIZE_512>(y, x, grid_size, params, _meta.unit(), cuda_stream);
        } else if (block_size <= CUDA_BLOCK_SIZE_1024) {
            status = launchKernel<CUDA_BLOCK_SIZE_1024>(y, x, grid_size, params, _meta.unit(), cuda_stream);
        } else {
            return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
        }
    }

    return status;
}

} // namespace op::rearrange::nvidia
