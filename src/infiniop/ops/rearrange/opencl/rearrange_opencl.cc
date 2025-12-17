#include "rearrange_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../../../tensor.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <vector>

namespace op::rearrange::opencl {

using ARRAY_TYPE_STRIDE = ptrdiff_t;
using ARRAY_TYPE_SIZE = size_t;

template <typename ElementType>
struct Constraint {
    ElementType grid_idx;
    ElementType block_idx;
    ElementType grid_div_block;
    ElementType total_len;
};

// RearrangeParams
struct RearrangeParams {
    std::vector<ARRAY_TYPE_SIZE> block_len;
    std::vector<ARRAY_TYPE_STRIDE> src_block_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_block_stride;
    std::vector<ARRAY_TYPE_SIZE> grid_len;
    std::vector<ARRAY_TYPE_STRIDE> src_grid_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_grid_stride;
    size_t block_dim;
    size_t block_len_total;
    std::vector<Constraint<ARRAY_TYPE_SIZE>> constraints;
    size_t unit_size; // bytes per unit
};

inline std::string unitSizeToOpenCLType(size_t unit_size) {
    switch (unit_size) {
    case 1:
        return "uchar";
    case 2:
        return "uchar2";
    case 4:
        return "float";
    case 8:
        return "float2";
    case 16:
        return "float4";
    case 32:
        return "double4"; // 需要 cl_khr_fp64
    default:
        return ""; // unsupported
    }
}

inline std::string kernelName(const RearrangeParams &p, int constraint_num) {
    std::ostringstream oss;
    oss << "rearrange_unit_" << unitSizeToOpenCLType(p.unit_size)
        << "_block_" << p.block_len.size()
        << "_grid_" << p.grid_len.size()
        << "_constrain_" << constraint_num;
    return oss.str();
}

inline std::string generateOpenCLKernel(const RearrangeParams &p) {
    auto grid_num = p.grid_len.size();
    auto block_num = p.block_len.size();
    auto constraint_num = p.constraints.size();
    CHECK_OR_RETURN(grid_num <= 5 && grid_num != 0, NULL);
    CHECK_OR_RETURN(block_num <= 5 && block_num != 0, NULL); // grid和block的维数都不超过5
    CHECK_OR_RETURN(constraint_num <= 2, NULL);
    auto unit_type = unitSizeToOpenCLType(p.unit_size);
    if (unit_type.empty()) {
        throw std::runtime_error("unsupported unit_size");
    }

    const size_t B = p.block_len.size();
    const size_t G = p.grid_len.size();

    std::ostringstream s;

    s << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"; // for double4 type
    s << "typedef int ARRAY_TYPE_STRIDE;\n";
    s << "typedef unsigned int ARRAY_TYPE_SIZE;\n\n";

    std::string kn = kernelName(p, constraint_num);
    s << "// Generated kernel: " << kn << "\n";

    s << "__kernel void " << kn << "(\n";
    s << "    __global uchar* dst,\n";
    s << "    __global const uchar* src,\n";
    s << "    const ulong block_len_total";

    for (size_t i = 0; i < B; ++i) {
        s << ",\n    const ARRAY_TYPE_SIZE block_len_" << i;
        s << ",\n    const ARRAY_TYPE_STRIDE src_block_stride_" << i;
        s << ",\n    const ARRAY_TYPE_STRIDE dst_block_stride_" << i;
    }

    for (size_t i = 0; i < G; ++i) {
        s << ",\n    const ARRAY_TYPE_SIZE grid_len_" << i;
        s << ",\n    const ARRAY_TYPE_STRIDE src_grid_stride_" << i;
        s << ",\n    const ARRAY_TYPE_STRIDE dst_grid_stride_" << i;
    }

    for (int c = 0; c < constraint_num; ++c) {
        s << ",\n    const ARRAY_TYPE_SIZE c" << c << "_grid_idx";
        s << ",\n    const ARRAY_TYPE_SIZE c" << c << "_block_idx";
        s << ",\n    const ARRAY_TYPE_SIZE c" << c << "_grid_div_block";
        s << ",\n    const ARRAY_TYPE_SIZE c" << c << "_total_len";
    }

    s << " ) {\n";
    s << "    const size_t local_tid = get_local_id(0);\n";
    s << "    const size_t group_id = get_group_id(0);\n\n";

    s << "    if (local_tid >= (size_t)block_len_total) return;\n\n";

    s << "    __local ARRAY_TYPE_STRIDE shared_src_offset_arr[1];\n";
    s << "    __local ARRAY_TYPE_STRIDE shared_dst_offset_arr[1];\n";
    if (constraint_num > 0) {
        s << "    __local ARRAY_TYPE_SIZE shared_constraints_grid_idx_multiple[" << constraint_num << "];\n";
    }
    s << "\n";

    s << "    if (local_tid == 0) {\n";
    s << "        ARRAY_TYPE_STRIDE src_offset = 0;\n";
    s << "        ARRAY_TYPE_STRIDE dst_offset = 0;\n";
    if (constraint_num > 0) {
        s << "        ARRAY_TYPE_SIZE constraints_grid_idx_multiple[" << constraint_num << "];\n";
    }
    s << "        size_t rem = group_id;\n\n";

    for (int i = (int)G - 1; i >= 0; --i) {
        s << "        {\n";
        s << "            size_t idx = rem % (size_t)grid_len_" << i << ";\n";
        s << "            rem = rem / (size_t)grid_len_" << i << ";\n";
        s << "            src_offset += (ARRAY_TYPE_STRIDE)idx * src_grid_stride_" << i << ";\n";
        s << "            dst_offset += (ARRAY_TYPE_STRIDE)idx * dst_grid_stride_" << i << ";\n";
        if (constraint_num > 0) {
            for (int j = 0; j < constraint_num; ++j) {
                s << "            if (" << i << " == (int)c" << j << "_grid_idx) constraints_grid_idx_multiple[" << j << "] = idx * (ARRAY_TYPE_SIZE)c" << j << "_grid_div_block;\n";
            }
        }
        s << "        }\n";
    }

    s << "        shared_src_offset_arr[0] = src_offset;\n";
    s << "        shared_dst_offset_arr[0] = dst_offset;\n";
    if (constraint_num > 0) {
        s << "        for (int j=0; j<" << constraint_num << "; ++j) shared_constraints_grid_idx_multiple[j] = constraints_grid_idx_multiple[j];\n";
    }
    s << "    }\n\n";

    // barrier and load
    s << "    barrier(CLK_LOCAL_MEM_FENCE);\n\n";
    s << "    ARRAY_TYPE_STRIDE src_offset = shared_src_offset_arr[0];\n";
    s << "    ARRAY_TYPE_STRIDE dst_offset = shared_dst_offset_arr[0];\n";
    if (constraint_num > 0) {
        s << "    ARRAY_TYPE_SIZE constraints_grid_idx_multiple[" << constraint_num << "];\n";
        s << "    for (int j=0;j<" << constraint_num << "; ++j) constraints_grid_idx_multiple[j] = shared_constraints_grid_idx_multiple[j];\n";
    }
    s << "\n";
    s << "    size_t rem_local = local_tid;\n\n";
    for (int i = (int)B - 1; i >= 1; --i) {
        s << "    {\n";
        s << "        size_t idx = rem_local % (size_t)block_len_" << i << ";\n";
        s << "        rem_local = rem_local / (size_t)block_len_" << i << ";\n";
        s << "        src_offset += (ARRAY_TYPE_STRIDE)idx * src_block_stride_" << i << ";\n";
        s << "        dst_offset += (ARRAY_TYPE_STRIDE)idx * dst_block_stride_" << i << ";\n";
        if (constraint_num > 0) {
            for (int j = 0; j < constraint_num; ++j) {
                s << "        if (" << i << " == (int)c" << j << "_block_idx) { if (constraints_grid_idx_multiple[" << j << "] + idx >= c" << j << "_total_len) return; }\n";
            }
        }
        s << "    }\n";
    }

    s << "    {\n";
    s << "        size_t idx = rem_local;\n";
    s << "        src_offset += (ARRAY_TYPE_STRIDE)idx * src_block_stride_0;\n";
    s << "        dst_offset += (ARRAY_TYPE_STRIDE)idx * dst_block_stride_0;\n";
    if (constraint_num > 0) {
        for (int j = 0; j < constraint_num; ++j) {
            s << "        if (0 == (int)c" << j << "_block_idx) { if (constraints_grid_idx_multiple[" << j << "] + idx >= c" << j << "_total_len) return; }\n";
        }
    }
    s << "    }\n\n";

    s << "    // unit_size = " << p.unit_size << " (bytes)\n";
    if (p.unit_size == 1) {
        s << "    dst[dst_offset] = src[src_offset];\n";
    } else {
        s << "    for (size_t b = 0; b < " << p.unit_size << "; ++b) {\n";
        s << "        dst[dst_offset + b] = src[src_offset + b];\n";
        s << "    }\n";
    }

    s << "}\n";

    return s.str();
}

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
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
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

struct Dim {
    size_t len;
    ARRAY_TYPE_STRIDE src_stride;
    ARRAY_TYPE_STRIDE dst_stride;
};

struct SplitDim {
    size_t choose_idx;
    size_t num_per_block;
    size_t num_per_grid;
    int array_struct_idx_block;
    int array_struct_idx_grid;
    size_t dim_len;
};

utils::Result<RearrangeParams> prepareRearrangeParams(const utils::RearrangeMeta &original_meta, int max_threads) {
    RearrangeParams params;

    auto meta_result = original_meta.distributeUnit({32, 16, 8, 4, 2, 1});

    CHECK_RESULT(meta_result);

    const utils::RearrangeMeta &meta = meta_result.take();

    const size_t ndim = meta.ndim();
    const size_t unit = meta.unit();

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

    const ptrdiff_t *idx_strides = meta.idx_strides();
    const ptrdiff_t *dst_strides = meta.dst_strides();
    const ptrdiff_t *src_strides = meta.src_strides();

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

    std::vector<size_t> src_strides_desc_idx(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        src_strides_desc_idx[i] = i;
    }
    std::sort(src_strides_desc_idx.begin(), src_strides_desc_idx.end(),
              [&dims](size_t a, size_t b) {
                  return std::abs(dims[a].src_stride) > std::abs(dims[b].src_stride);
              });

    const size_t block_size = max_threads;
    std::vector<bool> block_dim_choose(ndim, false);

    size_t block_elements = 1;
    size_t block_src_elements = 1;
    size_t block_dst_elements = 1;
    size_t src_choose_idx = ndim;
    size_t dst_choose_idx = ndim;

    std::vector<SplitDim> split_dims;

    while (src_choose_idx > 0 && dst_choose_idx > 0) {
        size_t src_idx = src_strides_desc_idx[src_choose_idx - 1];
        size_t dst_idx = dst_choose_idx - 1;

        if (src_idx == dst_idx) {
            size_t idx = src_idx;
            size_t len = shape[idx];

            if (block_elements * len <= block_size) {
                block_dim_choose[idx] = true;
                block_elements *= len;
                block_src_elements *= len;
                block_dst_elements *= len;
                src_choose_idx--;
                dst_choose_idx--;
            } else {

                size_t num_per_block = block_size / block_elements;

                if (num_per_block > 0 && len >= num_per_block && num_per_block > 1) {
                    size_t num_per_grid = (len + num_per_block - 1) / num_per_block;

                    SplitDim split_dim = {
                        idx,
                        num_per_block,
                        num_per_grid,
                        0,
                        0,
                        len};
                    split_dims.push_back(split_dim);
                }
                break;
            }
        } else {
            double src_div_dst = static_cast<double>(block_src_elements) / block_dst_elements;
            double src_num_per_block = std::sqrt(block_size / (double)block_elements / src_div_dst);
            double dst_num_per_block = src_num_per_block * src_div_dst;

            size_t src_current_dim_len = shape[src_idx];
            size_t dst_current_dim_len = shape[dst_idx];

            if (static_cast<double>(src_current_dim_len) < src_num_per_block) {
                block_dim_choose[src_idx] = true;
                block_elements *= src_current_dim_len;
                block_src_elements *= src_current_dim_len;
                src_choose_idx--;
            } else if (static_cast<double>(dst_current_dim_len) < dst_num_per_block) {
                block_dim_choose[dst_idx] = true;
                block_elements *= dst_current_dim_len;
                block_dst_elements *= dst_current_dim_len;
                dst_choose_idx--;
            } else {
                size_t src_num_per_block_int = static_cast<size_t>(std::floor(src_num_per_block));
                size_t dst_num_per_block_int = static_cast<size_t>(std::floor(dst_num_per_block));

                size_t src_num_per_grid = (src_current_dim_len + src_num_per_block_int - 1) / src_num_per_block_int;
                size_t dst_num_per_grid = (dst_current_dim_len + dst_num_per_block_int - 1) / dst_num_per_block_int;

                if (src_num_per_block_int > 1) {
                    if (src_num_per_grid == 1) {

                        block_dim_choose[src_idx] = true;
                        block_elements *= src_current_dim_len;
                        block_src_elements *= src_current_dim_len;
                        src_choose_idx--;
                    } else {
                        SplitDim split_dim = {
                            src_idx,
                            src_num_per_block_int,
                            src_num_per_grid,
                            0,
                            0,
                            src_current_dim_len};
                        split_dims.push_back(split_dim);
                    }
                }

                if (dst_num_per_block_int > 1) {
                    if (dst_num_per_grid == 1) {

                        block_dim_choose[dst_idx] = true;
                        block_elements *= dst_current_dim_len;
                        block_dst_elements *= dst_current_dim_len;
                        dst_choose_idx--;
                    } else {

                        SplitDim split_dim = {
                            dst_idx,
                            dst_num_per_block_int,
                            dst_num_per_grid,
                            0,
                            0,
                            dst_current_dim_len};
                        split_dims.push_back(split_dim);
                    }
                }

                break;
            }
        }
    }

    size_t block_dim = 0;
    size_t block_len_total = 1;

    std::vector<ARRAY_TYPE_SIZE> block_len;
    std::vector<ARRAY_TYPE_STRIDE> src_block_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_block_stride;

    std::vector<ARRAY_TYPE_SIZE> grid_len;
    std::vector<ARRAY_TYPE_STRIDE> src_grid_stride;
    std::vector<ARRAY_TYPE_STRIDE> dst_grid_stride;

    for (size_t i = 0; i < ndim; ++i) {
        if (block_dim_choose[i]) {
            block_len.push_back(shape[i]);
            src_block_stride.push_back(dims[i].src_stride);
            dst_block_stride.push_back(dims[i].dst_stride);
            block_dim += 1;
            block_len_total *= shape[i];
        }

        for (size_t j = 0; j < split_dims.size(); ++j) {
            if (i == split_dims[j].choose_idx) {
                block_len.push_back(split_dims[j].num_per_block);
                src_block_stride.push_back(dims[i].src_stride);
                dst_block_stride.push_back(dims[i].dst_stride);
                split_dims[j].array_struct_idx_block = static_cast<int>(block_dim);
                block_dim += 1;
                block_len_total *= split_dims[j].num_per_block;
            }
        }
    }

    for (size_t i = 0; i < ndim; ++i) {
        if (!block_dim_choose[i]) {
            bool is_split = false;

            for (size_t j = 0; j < split_dims.size(); ++j) {
                if (i == split_dims[j].choose_idx) {
                    is_split = true;
                    grid_len.push_back(split_dims[j].num_per_grid);
                    src_grid_stride.push_back(dims[i].src_stride * split_dims[j].num_per_block);
                    dst_grid_stride.push_back(dims[i].dst_stride * split_dims[j].num_per_block);
                    split_dims[j].array_struct_idx_grid = static_cast<int>(grid_len.size() - 1);
                }
            }

            if (!is_split) {
                grid_len.push_back(shape[i]);
                src_grid_stride.push_back(dims[i].src_stride);
                dst_grid_stride.push_back(dims[i].dst_stride);
            }
        }
    }

    if (grid_len.empty()) {
        grid_len.push_back(1);
        src_grid_stride.push_back(0);
        dst_grid_stride.push_back(0);
    }

    std::vector<Constraint<ARRAY_TYPE_SIZE>> constraints;

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
    }

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

infiniStatus_t launchKernel(
    void *y,
    const void *x,
    size_t y_size, size_t x_size,
    const RearrangeParams &params,
    size_t grid_size, size_t block_size, size_t unit_size,
    cl_context context, cl_device_id device,
    cl_command_queue cl_queue, cl_program program) {

    cl_int clerr;
    cl_kernel kernel = clCreateKernel(program, kernelName(params, params.constraints.size()).c_str(), &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);

    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&y_svm, y_size);
        clerr = infinirtMemcpy(y_svm, y, y_size, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    void *x_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&x_svm, x_size);
        infinirtMemcpy(x_svm, x, x_size, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }
    clerr = clSetKernelArg(kernel, arg_idx++, sizeof(unsigned int), &block_size);
    if (clerr != CL_SUCCESS) {
        std::cerr << "[OpenCL] clSetKernelArg(block_size) failed\n";
    }

    for (size_t i = 0; i < params.block_len.size(); ++i) {
        ARRAY_TYPE_SIZE v_block_len = static_cast<ARRAY_TYPE_SIZE>(params.block_len[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v_block_len);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(block_len_" << i << ") failed\n";
        }

        ARRAY_TYPE_STRIDE v_src_bs = static_cast<ARRAY_TYPE_STRIDE>(params.src_block_stride[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_STRIDE), &v_src_bs);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(src_block_stride_" << i << ") failed\n";
        }

        ARRAY_TYPE_STRIDE v_dst_bs = static_cast<ARRAY_TYPE_STRIDE>(params.dst_block_stride[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_STRIDE), &v_dst_bs);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(dst_block_stride_" << i << ") failed\n";
        }
    }

    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        ARRAY_TYPE_SIZE v_grid_len = static_cast<ARRAY_TYPE_SIZE>(params.grid_len[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v_grid_len);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(grid_len_" << i << ") failed\n";
        }

        ARRAY_TYPE_STRIDE v_src_gs = static_cast<ARRAY_TYPE_STRIDE>(params.src_grid_stride[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_STRIDE), &v_src_gs);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(src_grid_stride_" << i << ") failed\n";
        }

        ARRAY_TYPE_STRIDE v_dst_gs = static_cast<ARRAY_TYPE_STRIDE>(params.dst_grid_stride[i]);
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_STRIDE), &v_dst_gs);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(dst_grid_stride_" << i << ") failed\n";
        }
    }

    for (size_t c = 0; c < params.constraints.size(); ++c) {
        const auto &cc = params.constraints[c];
        ARRAY_TYPE_SIZE v0 = cc.grid_idx;
        ARRAY_TYPE_SIZE v1 = cc.block_idx;
        ARRAY_TYPE_SIZE v2 = cc.grid_div_block;
        ARRAY_TYPE_SIZE v3 = cc.total_len;
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v0);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(c" << c << "_grid_idx) failed\n";
        }
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v1);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(c" << c << "_block_idx) failed\n";
        }
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v2);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(c" << c << "_grid_div_block) failed\n";
        }
        clerr = clSetKernelArg(kernel, arg_idx++, sizeof(ARRAY_TYPE_SIZE), &v3);
        if (clerr != CL_SUCCESS) {
            std::cerr << "[OpenCL] clSetKernelArg(c" << c << "_total_len) failed\n";
        }
    }

    size_t global_size[1] = {block_size * grid_size};
    size_t local_size[1] = {block_size};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    if (y_svm) { // for python test
        infinirtMemcpy(y, y_svm, y_size, INFINIRT_MEMCPY_D2H);
    }
    clReleaseKernel(kernel);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {

    void *device;
    void *context;
    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));
    cl_context clcontext = static_cast<cl_context>(context);
    cl_device_id cldevice = static_cast<cl_device_id>(device);
    if (!stream) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream));
    }
    cl_command_queue clqueue = static_cast<cl_command_queue>(stream);

    if (_meta.ndim() == 0) {
        auto clerr = infinirtMemcpyAsync(y, x, _meta.unit(), INFINIRT_MEMCPY_D2D, clqueue);

        if (clerr != CL_SUCCESS) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    }

    const size_t ndim = _meta.ndim();
    const size_t unit = _meta.unit();

    const ptrdiff_t *idx_strides = _meta.idx_strides();
    const ptrdiff_t *dst_strides = _meta.dst_strides();
    const ptrdiff_t *src_strides = _meta.src_strides();
    std::vector<size_t> shape;
    shape.reserve(ndim);

    auto prev_idx_stride = _meta.count();
    for (size_t i = 0; i < ndim; ++i) {
        size_t len = prev_idx_stride / idx_strides[i];
        shape.push_back(len);
        prev_idx_stride = idx_strides[i];
    }

    size_t y_size = unit;
    size_t x_size = unit;
    for (size_t i = 0; i < ndim; ++i) {
        y_size += dst_strides[i] * (shape[i] - 1);
        x_size += src_strides[i] * (shape[i] - 1);
    }

    int max_threads = _opaque->internal->maxThreadsPerBlock();

    auto params_result = prepareRearrangeParams(_meta, max_threads);
    CHECK_RESULT(params_result);
    auto params = params_result.take();

    size_t grid_size = 1;
    for (size_t i = 0; i < params.grid_len.size(); ++i) {
        grid_size *= params.grid_len[i];
    }

    if (grid_size == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    infiniStatus_t status = INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    size_t block_size = params.block_len_total;

    RearrangeParams params_copy = params;
    std::string RearrangeKernelSource = generateOpenCLKernel(params_copy);

    std::string build_opts;
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource(kernelName(params, params.constraints.size()).c_str(), RearrangeKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    status = launchKernel(y, x, y_size, x_size, params, grid_size, block_size, _meta.unit(), clcontext, cldevice, clqueue, clprogram);
    return status;
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::opencl
