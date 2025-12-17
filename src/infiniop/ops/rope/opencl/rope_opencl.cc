#include "rope_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <fstream>
#include <memory>
#include <sstream>

static const char *RopeKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float2
#endif

#ifndef Tpos
#define Tpos unsigned int
#endif

#ifndef Ttable
#define Ttable float
#endif

#ifdef USE_HALF
#define LOAD_DATA(ptr) vload_half2(0, (__global half *) ptr)
#define STORE_DATA(ptr, val) vstore_half2(val, 0, (__global half *) ptr)
#else
#define LOAD_DATA(ptr) (*ptr)
#define STORE_DATA(ptr, val) (*ptr = val)
#endif

typedef unsigned int Tidx;

__kernel void rope(
    __global Tval *t,
    int const ystride_token,
    int const ystride_head,
    __global Tval *x,
    int const xstride_token,
    int const xstride_head,
    __global Tpos const *pos,
    __global Ttable const *sin_table,
    __global Ttable const *cos_table,
    float const theta) {

    Tidx nh_l = get_local_size(0),
         dh = get_local_size(1),
         it = get_group_id(0),
         ih_h = get_group_id(1),
         ih_l = get_local_id(0),
         ih = ih_h * nh_l + ih_l,
         i = get_local_id(1);

    __global Tval *t2 = t + it * ystride_token + ih * ystride_head + i;
    __global Tval *x2 = x + it * xstride_token + ih * xstride_head + i;

    float2 data = LOAD_DATA(x2);
    
    int index = pos[it] * dh + i;  // 防越界
    float sin_val = sin_table[index];
    float cos_val = cos_table[index];

    float2 result;
    result.x = data.x * cos_val - data.y * sin_val;
    result.y = data.x * sin_val + data.y * cos_val;
    STORE_DATA(t2, result);
}

)CLC";

namespace op::rope::opencl {

using namespace device::opencl::kernel;

struct Descriptor::Opaque {
    std::shared_ptr<device::opencl::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t pos_desc,
    infiniopTensorDescriptor_t sin_desc,
    infiniopTensorDescriptor_t cos_desc,
    infiniopRoPEAlgo_t algo) {

    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);

    auto info = RoPEInfo::createRoPEInfo(y_desc, x_desc, pos_desc, sin_desc, cos_desc, algo);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::opencl::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    size_t dimx, size_t dimy, size_t table_dim,
    void *y, infiniDtype_t tdata, ptrdiff_t y_stride_seqlen, ptrdiff_t y_stride_nhead,
    const void *x, ptrdiff_t x_stride_seqlen, ptrdiff_t x_stride_nhead,
    infiniDtype_t tpos, const void *pos_ids, const void *sin_table, const void *cos_table,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue,
    cl_program program) {

    cl_int clerr;
    cl_kernel kernel = clCreateKernel(program, "rope", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *y_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&y_svm, ((dimx - 1) * y_stride_seqlen + (dimy - 1) * y_stride_nhead + table_dim) * dtypeSize(tdata));
        infinirtMemcpy(y_svm, y, ((dimx - 1) * y_stride_seqlen + (dimy - 1) * y_stride_nhead + table_dim) * dtypeSize(tdata), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, y_svm);
    }
    cl_int s_y_batch = static_cast<cl_int>(y_stride_seqlen) / 2;
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_batch);
    cl_int s_y_nhead = static_cast<cl_int>(y_stride_nhead) / 2;
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_y_nhead);
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, x);
    if (clerr != CL_SUCCESS) { // for python test
        void *x_svm = NULL;
        infinirtMalloc(&x_svm, ((dimx - 1) * x_stride_seqlen + (dimy - 1) * x_stride_nhead + table_dim * 2) * dtypeSize(tdata));
        infinirtMemcpy(x_svm, x, ((dimx - 1) * x_stride_seqlen + (dimy - 1) * x_stride_nhead + table_dim * 2) * dtypeSize(tdata), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, x_svm);
    }
    cl_int s_x_batch = static_cast<cl_int>(x_stride_seqlen) / 2;
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_batch);
    cl_int s_x_nhead = static_cast<cl_int>(x_stride_nhead) / 2;
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &s_x_nhead);

    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, pos_ids);
    if (clerr != CL_SUCCESS) { // for python test
        void *pos_svm = NULL;
        infinirtMalloc(&pos_svm, dimx * dtypeSize(tpos));
        infinirtMemcpy(pos_svm, pos_ids, dimx * dtypeSize(tpos), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, pos_svm);
    }
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, sin_table);
    if (clerr != CL_SUCCESS) { // for python test
        void *sin_svm = NULL;
        infinirtMalloc(&sin_svm, dimx * table_dim * dtypeSize(tdata));
        infinirtMemcpy(sin_svm, sin_table, dimx * table_dim * dtypeSize(tdata), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, sin_svm);
    }
    clerr |= clSetKernelArgSVMPointer(kernel, arg_idx++, cos_table);
    if (clerr != CL_SUCCESS) { // for python test
        void *cos_svm = NULL;
        infinirtMalloc(&cos_svm, dimx * table_dim * dtypeSize(tdata));
        infinirtMemcpy(cos_svm, cos_table, dimx * table_dim * dtypeSize(tdata), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, cos_svm);
    }
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(int), &table_dim);

    if (block_size % table_dim != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    int max_nh_l = std::min(block_size / table_dim, dimy);
    int nh_l = 1;
    for (int candidate = max_nh_l; candidate >= 1; --candidate) {
        if (dimy % candidate == 0) {
            nh_l = candidate;
            break;
        }
    }
    int nh_h = dimy / nh_l;
    size_t global_size[2] = {dimx * nh_l, nh_h * table_dim};
    size_t local_size[2] = {nh_l, table_dim};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel);

        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (y_svm) { // for python test
        infinirtMemcpy(y, y_svm, ((dimx - 1) * y_stride_seqlen + (dimy - 1) * y_stride_nhead + table_dim * 2) * dtypeSize(tdata), INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto y_stride_seqlen = _info.y_stride_seqlen;
    auto y_stride_nhead = _info.y_stride_nhead;
    auto x_stride_seqlen = _info.x_stride_seqlen;
    auto x_stride_nhead = _info.x_stride_nhead;
    auto table_dim = _info.table_dim;

    auto dimx = uint32_t(_info.seqlen),
         dimy = uint32_t(_info.nhead);
    size_t block_size = _opaque->internal->maxThreadsPerBlock();
    auto tdata = _info.data_type;
    auto tpos = _info.pos_type;

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

    std::string dt_val, dt_pos, dt_table;
    if (tdata == INFINI_DTYPE_F16) {
        dt_val = "half2";
        dt_table = "half";
    } else if (tdata == INFINI_DTYPE_F32) {
        dt_val = "float2";
        dt_table = "float";
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (tpos == INFINI_DTYPE_I32) {
        dt_pos = "int";
    } else if (tpos == INFINI_DTYPE_U32) {
        dt_pos = "uint";
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Tval=" + dt_val + " ";
    build_opts += "-D Tpos=" + dt_pos + " ";
    build_opts += "-D Ttable=" + dt_table + " ";
    if (tdata == INFINI_DTYPE_F16) {
        build_opts += "-D USE_HALF=1 ";
    }
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource("rope", RopeKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(dimx, dimy, table_dim, y, tdata, y_stride_seqlen, y_stride_nhead,
                              x, x_stride_seqlen, x_stride_nhead, tpos, pos_ids, sin_table, cos_table, block_size, clcontext, cldevice, clqueue, clprogram));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rope::opencl
