#include "rms_norm_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../../../devices/opencl/opencl_kernel_common.h"
#include <fstream>
#include <memory>
#include <sstream>

static const char *RmsNormKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Ta
#define Ta float
#endif

#ifndef Tw
#define Tw float
#endif

#ifndef Tcompute
#define Tcompute float
#endif

#ifndef ITEMS_THREAD
#define ITEMS_THREAD 1
#endif

typedef unsigned int Tidx;

kernel void rms_norm(
    global Ta *y_,
    int const s_y_batch,
    int const s_y_nhead,
    global Ta const *x_,
    int const s_x_batch,
    int const s_x_nhead,
    global Tw const *w,
    float const epsilon,
    Tidx const nhead,
    Tidx const d) {

    Tidx g_idx = get_group_id(0),
         l_idx = get_local_id(0),
         l_len = get_local_size(0);
    Tidx batch_id = g_idx / nhead,
         nhead_id = g_idx % nhead;
    global Ta
        *y = y_ + batch_id * s_y_batch + nhead_id * s_y_nhead;
    global Ta const
        *x = x_ + batch_id * s_x_batch + nhead_id * s_x_nhead;

    Tcompute val_x[ITEMS_THREAD];
    Tcompute val_w[ITEMS_THREAD];
    Tcompute squared = 0;
    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len) {
        val_x[i] = (Tcompute)x[idx];
        val_w[i] = (Tcompute)w[idx];
        squared += val_x[i] * val_x[i];
    }
    // TODO:测试加载相邻元素处理；
    Tcompute mean_sq = work_group_reduce_add(squared) / (Tcompute)d;
    Tcompute rms = native_rsqrt(mean_sq + (Tcompute)epsilon);

    for (Tidx i = 0, idx = l_idx; idx < d; ++i, idx += l_len)
        y[idx] = (Ta)(rms * val_x[i] * val_w[i]);
}
)CLC";

namespace op::rms_norm::opencl {

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
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// launch kernel
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t nhead, size_t dim,
    void *y, infiniDtype_t atype, ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    const void *x, ptrdiff_t stride_x_batch, ptrdiff_t stride_x_nhead,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue,
    cl_program program) {

    cl_int clerr;

    cl_kernel kernel = clCreateKernel(program, "rms_norm", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&y_svm, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * infiniSizeOf(atype));
        infinirtMemcpy(y_svm, y, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * infiniSizeOf(atype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    cl_int s_y_batch = static_cast<cl_int>(stride_y_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_batch);
    cl_int s_y_nhead = static_cast<cl_int>(stride_y_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_nhead);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) { // for python test
        void *x_svm = NULL;
        infinirtMalloc(&x_svm, ((batch_size - 1) * stride_x_batch + (nhead - 1) * stride_x_nhead + dim) * infiniSizeOf(atype));
        infinirtMemcpy(x_svm, x, ((batch_size - 1) * stride_x_batch + (nhead - 1) * stride_x_nhead + dim) * infiniSizeOf(atype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }
    cl_int s_x_batch = static_cast<cl_int>(stride_x_batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_batch);
    cl_int s_x_nhead = static_cast<cl_int>(stride_x_nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_nhead);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, w);
    if (clerr != CL_SUCCESS) { // for python test
        void *w_svm = NULL;
        infinirtMalloc(&w_svm, dim * infiniSizeOf(wtype));
        infinirtMemcpy(w_svm, w, dim * infiniSizeOf(wtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, w_svm);
    }
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &epsilon);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &nhead);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &dim);

    size_t global_size = batch_size * nhead * block_size;

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 1, nullptr, &global_size, &block_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size, block_size);
        clReleaseKernel(kernel);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (y_svm) { // for python test
        infinirtMemcpy(y, y_svm, ((batch_size - 1) * stride_y_batch + (nhead - 1) * stride_y_nhead + dim) * infiniSizeOf(atype), INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_x_batch = _info.x_strides[0];
    auto stride_x_nhead = _info.x_strides[1];
    auto stride_y_batch = _info.y_strides[0];
    auto stride_y_nhead = _info.y_strides[1];
    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    size_t nhead = _info.shape.size() > 2 ? _info.shape[1] : 1;
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

    std::string dt_a, dt_w, dt_compute;
    dt_compute = "float";
    if (!dtypeToClType(_info.atype, dt_a)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtypeToClType(_info.wtype, dt_w)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    size_t items_perthread = (dim + block_size - 1) / block_size;

    // build options
    std::string build_opts;
    build_opts += "-D Ta=" + dt_a + " ";
    build_opts += "-D Tw=" + dt_w + " ";
    build_opts += "-D Tc=" + dt_compute + " ";
    build_opts += "-D ITEMS_THREAD=" + std::to_string(items_perthread) + " ";
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource("rms_norm", RmsNormKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, block_size, clcontext, cldevice, clqueue, clprogram));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::opencl
