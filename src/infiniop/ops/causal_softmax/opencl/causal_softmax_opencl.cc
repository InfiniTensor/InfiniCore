#include "causal_softmax_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <fstream>
#include <memory>
#include <sstream>

const size_t ITEMS_THREAD = 8;

static const char *CausalSoftmaxKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 8
#endif

#ifndef MASK
#define MASK causal_mask
#endif

typedef unsigned int Tidx;

bool causal_mask(Tidx tok_id, Tidx seq_len,
                 Tidx pos_id, Tidx att_len) {
    //   tok_id ↓ |<---att_len--->|
    //          0 | * * ... *     |
    //          1 | * * ... * *   |
    //          2 | * * ... * * * |
    // seq_len: 3 |---------------|
    return att_len + tok_id >= pos_id + seq_len;
}

kernel void softmax_register(
    global Tval *att_y,
    global Tval *att_x,
    Tidx const seq_len,
    Tidx const att_len,
    int const head_stride_y,
    int const tok_stride_y,
    int const head_stride_x,
    int const tok_stride_x) {

    Tidx const
        head_idx = get_group_id(1),
        tok_id = get_group_id(0),
        l_idx = get_local_id(0),
        l_len = get_local_size(0);

    global Tval *y = att_y + head_idx * head_stride_y + tok_id * tok_stride_y;
    global Tval *x = att_x + head_idx * head_stride_x + tok_id * tok_stride_x;

    float
        data[ITEMS_THREAD],
        max_ = -FLT_MAX,
        sum_ = 0;

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        data[i] = causal_mask(tok_id, seq_len, idx, att_len) ? x[idx] : -FLT_MAX;
        max_ = fmax(max_, data[i]);
    }

    max_ = work_group_reduce_max(max_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        data[i] = exp(data[i] - max_);
        sum_ += data[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float const k = 1 / work_group_reduce_add(sum_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len)
        y[idx] = data[i] * k;
}

kernel void softmax_global(
    global Tval *att_y,
    global Tval *att_x,
    Tidx const seq_len,
    Tidx const att_len,
    int const head_stride_y,
    int const tok_stride_y,
    int const head_stride_x,
    int const tok_stride_x) {

    Tidx const
        head_idx = get_group_id(1),
        tok_id = get_group_id(0),
        l_idx = get_local_id(0),
        l_len = get_local_size(0);

    global Tval *y = att_y + head_idx * head_stride_y + tok_id * tok_stride_y;
    global Tval *x = att_x + head_idx * head_stride_x + tok_id * tok_stride_x;

    float
        max_ = -FLT_MAX,
        sum_ = 0;

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        float const data = causal_mask(tok_id, seq_len, idx, att_len) ? x[idx] : -FLT_MAX;
        max_ = fmax(max_, data);
    }

    max_ = work_group_reduce_max(max_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len) {
        float const data = exp(x[idx] - max_);
        y[idx] = data;
        sum_ += data;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    float const k = 1 / work_group_reduce_add(sum_);

    for (Tidx i = 0, idx = l_idx; idx < att_len; ++i, idx += l_len)
        y[idx] *= k;
}

)CLC";

inline int last_power_of_two(int n) {
    int p = 1;
    while (p * 2 <= n) {
        p *= 2;
    }
    return p;
}

namespace op::causal_softmax::opencl {

using namespace device::opencl::kernel;

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
    auto info = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(info);
    auto opaque = new Descriptor::Opaque{
        reinterpret_cast<device::opencl::Handle *>(handle)->internal()};
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        // opaque,
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t batch_size, size_t seq_len, size_t total_seq_len,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_i,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_i,
                            size_t block_size, cl_context context,
                            cl_device_id device, cl_command_queue cl_queue,
                            cl_program program) {
    cl_int clerr;

    int group_size = last_power_of_two(std::min(total_seq_len, total_seq_len));
    int items_thread = (total_seq_len + group_size - 1) / group_size;
    cl_kernel kernel;
    if (items_thread <= ITEMS_THREAD) {
        kernel = clCreateKernel(program, "softmax_register", &clerr);
        if (clerr != CL_SUCCESS || kernel == nullptr) {
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    } else {
        kernel = clCreateKernel(program, "softmax_global", &clerr);
        if (clerr != CL_SUCCESS || kernel == nullptr) {
            clReleaseProgram(program);
            return INFINI_STATUS_INTERNAL_ERROR;
        }
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&y_svm, ((batch_size - 1) * y_stride_b + (seq_len - 1) * y_stride_i + total_seq_len) * dtypeSize(dtype));
        infinirtMemcpy(y_svm, y, ((batch_size - 1) * y_stride_b + (seq_len - 1) * y_stride_i + total_seq_len) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) { // for python test
        void *x_svm = NULL;
        infinirtMalloc(&x_svm, ((batch_size - 1) * x_stride_b + (seq_len - 1) * x_stride_i + total_seq_len) * dtypeSize(dtype));
        infinirtMemcpy(x_svm, x, ((batch_size - 1) * x_stride_b + (seq_len - 1) * x_stride_i + total_seq_len) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }

    cl_int s_len = static_cast<cl_int>(seq_len);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_len);
    cl_int att_len = static_cast<cl_int>(total_seq_len);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &att_len);
    cl_int y_s_b = static_cast<cl_int>(y_stride_b);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &y_s_b);
    cl_int y_s_i = static_cast<cl_int>(y_stride_i);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &y_s_i);
    cl_int x_s_b = static_cast<cl_int>(x_stride_b);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &x_s_b);
    cl_int x_s_i = static_cast<cl_int>(x_stride_i);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &x_s_i);

    size_t global_size[2] = {group_size * seq_len, batch_size};
    size_t local_size[2] = {size_t(group_size), size_t(1)};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);

    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (y_svm) { // for python test
        infinirtMemcpy(y, y_svm, ((batch_size - 1) * y_stride_b + (seq_len - 1) * y_stride_i + total_seq_len) * dtypeSize(dtype), INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    size_t block_size = _opaque->internal->maxThreadsPerBlock();
    void *device;
    void *context;
    CHECK_STATUS(infinirtGetOpenclDevice(&device));
    CHECK_STATUS(infinirtGetOpenclContext(&context));
    cl_context clcontext = static_cast<cl_context>(context);
    cl_device_id cldevice = static_cast<cl_device_id>(device);
    if (!stream_) {
        CHECK_STATUS(infinirtGetOpenclStream(&stream_));
    }
    cl_command_queue clqueue = static_cast<cl_command_queue>(stream_);
    auto dtype = _info.dtype;
    std::string dt_val;
    if (dtype == INFINI_DTYPE_F16) {
        dt_val = "half";
    } else if (dtype == INFINI_DTYPE_F32) {
        dt_val = "float";
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Tval=" + dt_val + " ";
    build_opts += "-D ITEMS_THREAD=" + std::to_string(ITEMS_THREAD) + " ";
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource("causal_softmax", CausalSoftmaxKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(y, x, dtype, _info.batch_size, _info.seq_len, _info.total_seq_len,
                              _info.y_stride_b, _info.y_stride_i, _info.x_stride_b, _info.x_stride_i, block_size, clcontext, cldevice, clqueue, clprogram));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::opencl
