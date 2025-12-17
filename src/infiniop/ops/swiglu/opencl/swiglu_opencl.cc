#include "swiglu_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <fstream>
#include <math.h>
#include <memory>
#include <sstream>

static const char *SwigluKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

typedef unsigned int Tidx;

__kernel void swiglu(
    __global Tval *c,
    int const stride_c_b,
    int const stride_c_s,
    __global Tval *gate,
    int const stride_gate_b,
    int const stride_gate_s,
    __global Tval *up,
    int const stride_up_b,
    int const stride_up_s,
    int const seq) {

    Tidx g_idx = get_global_id(0);
    Tidx g_idy = get_global_id(1);
    Tidx g_idx_b = get_global_id(0) / seq;
    Tidx g_idx_s = get_global_id(0) % seq;

    Tidx k = g_idx_b * stride_c_b + g_idx_s * stride_c_s + g_idy;
    Tidx i = g_idx_b * stride_gate_b + g_idx_s * stride_gate_s + g_idy;
    Tidx j = g_idx_b * stride_up_b + g_idx_s * stride_up_s + g_idy;

    Tval x = gate[i];
    Tval y = up[j];

    Tval sig = 1.0f / (1.0f + exp(-x));
    c[k] = x * sig * y;
}

)CLC";

namespace op::swiglu::opencl {

using namespace device::opencl::kernel;

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  std::vector<infiniopTensorDescriptor_t> input_descs) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);

    auto dtype = c_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    const auto &a_desc = input_descs[0];
    const auto &b_desc = input_descs[1];

    auto info = SwigluInfo::create(c_desc, a_desc, b_desc);
    CHECK_RESULT(info);

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        info.take(),
        workspace_size,
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    void *c, void *a, void *b,
    infiniDtype_t dtype, size_t batch, size_t seq, size_t hd,
    ptrdiff_t stride_batch_c, ptrdiff_t stride_batch_a, ptrdiff_t stride_batch_b,
    ptrdiff_t stride_seq_c, ptrdiff_t stride_seq_a, ptrdiff_t stride_seq_b,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue,
    cl_program program) {

    cl_int clerr;
    cl_kernel kernel = clCreateKernel(program, "swiglu", &clerr);

    if (clerr != CL_SUCCESS || kernel == nullptr) {

        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *c_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, c);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&c_svm, ((batch - 1) * stride_batch_c + (seq - 1) * stride_seq_c + hd) * dtypeSize(dtype));
        infinirtMemcpy(c_svm, c, ((batch - 1) * stride_batch_c + (seq - 1) * stride_seq_c + hd) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, c_svm);
    }
    cl_int s_c_batch = static_cast<cl_int>(stride_batch_c);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_c_batch);
    cl_int s_c_seq = static_cast<cl_int>(stride_seq_c);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_c_seq);

    void *b_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&b_svm, ((batch - 1) * stride_batch_b + (seq - 1) * stride_seq_b + hd) * dtypeSize(dtype));
        infinirtMemcpy(b_svm, b, ((batch - 1) * stride_batch_b + (seq - 1) * stride_seq_b + hd) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_svm);
    }
    cl_int s_b_batch = static_cast<cl_int>(stride_batch_b);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_b_batch);
    cl_int s_b_seq = static_cast<cl_int>(stride_seq_b);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_b_seq);

    void *a_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&a_svm, ((batch - 1) * stride_batch_a + (seq - 1) * stride_seq_a + hd) * dtypeSize(dtype));
        infinirtMemcpy(a_svm, a, ((batch - 1) * stride_batch_a + (seq - 1) * stride_seq_a + hd) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_svm);
    }
    cl_int s_a_batch = static_cast<cl_int>(stride_batch_a);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_a_batch);
    cl_int s_a_seq = static_cast<cl_int>(stride_seq_a);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_a_seq);

    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &seq);

    size_t global_size[2] = {batch * seq, hd};
    size_t local_size[2] = {1, block_size};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (c_svm) { // for python test
        infinirtMemcpy(c, c_svm, ((batch - 1) * stride_batch_c + (seq - 1) * stride_seq_c + hd) * dtypeSize(dtype), INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     std::vector<const void *> inputs,
                                     void *stream) const {
    auto batch = _info.ndim == 2 ? 1 : _info.shape[0];
    auto seq_len = _info.ndim == 2 ? _info.shape[0] : _info.shape[1];
    auto hidden_size = _info.shape[_info.ndim - 1];
    auto stride_batch_c = _info.ndim == 2 ? 1 : _info.c_strides[0];
    auto stride_batch_a = _info.ndim == 2 ? 1 : _info.a_strides[0];
    auto stride_batch_b = _info.ndim == 2 ? 1 : _info.b_strides[0];
    auto stride_seq_c = _info.ndim == 2 ? _info.c_strides[0] : _info.c_strides[1];
    auto stride_seq_a = _info.ndim == 2 ? _info.a_strides[0] : _info.a_strides[1];
    auto stride_seq_b = _info.ndim == 2 ? _info.b_strides[0] : _info.b_strides[1];
    size_t block_size = _opaque->internal->maxThreadsPerBlock();
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

    std::string dt_val;
    if (_info.dtype == INFINI_DTYPE_F16) {
        dt_val = "half";
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        dt_val = "float";
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Tval=" + dt_val + " ";
    if (_info.dtype == INFINI_DTYPE_F16) {
        build_opts += "-D USE_HALF=1 ";
    }
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource("swiglu", SwigluKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(c, (void *)inputs[0], (void *)inputs[1], _info.dtype, batch, seq_len, hidden_size, stride_batch_c, stride_batch_a, stride_batch_b, stride_seq_c, stride_seq_a, stride_seq_b, block_size, clcontext, cldevice, clqueue, clprogram));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::swiglu::opencl
