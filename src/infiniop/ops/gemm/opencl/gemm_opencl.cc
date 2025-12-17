#include "gemm_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include <fstream>
#include <math.h>
#include <memory>
#include <sstream>

static const char *GemmKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif

#ifdef USE_HALF
#define MUL(valueA, valueB) (float)(valueA * valueB)
#define SCAL(beta, p, alpha, value) (half)(beta * (float)(*p) + alpha * value)
#define SCAL1(alpha, value) (half)(alpha * value)
#else
#define MUL(valueA, valueB) valueA *valueB
#define SCAL(beta, p, alpha, value) beta *(*p) + alpha *value
#define SCAL1(alpha, value) (half)(alpha * value)
#endif

__kernel void general_gemm(__global Tval *A, __global Tval *B, __global Tval *C,
                           int as, int ars, int acs, int bs, int brs, int bcs,
                           int cs, int crs, int ccs, int batch,
                           int M, int N, int K, float alpha, float beta) {
    int g_idx = get_global_id(1);
    int g_idy = get_global_id(0);
    int row_id = g_idy / N;
    int col_id = g_idy % N;

    Tval valueA = 0.0f;
    Tval valueB = 0.0f;
    float value = 0.0f;

    for (int i = 0; i < K; i++) {
        valueA = *(A + g_idx * as + row_id * ars + i * acs);
        valueB = *(B + g_idx * bs + i * brs + col_id * bcs);
        value += MUL(valueA, valueB);
    }

    __global Tval *p = C + g_idx * cs + row_id * crs + col_id * ccs;
    if (beta != 0)
        *p = SCAL(beta, p, alpha, value);
    else
        *p = SCAL1(alpha, value);
}

)CLC";

namespace op::gemm::opencl {

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
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launchKernel(
    void *c, const void *a, const void *b,
    infiniDtype_t dtype, size_t batch, size_t m, size_t n, size_t k, float alpha, float beta,
    ptrdiff_t c_stride, ptrdiff_t c_row_stride, ptrdiff_t c_col_stride,
    ptrdiff_t a_stride, ptrdiff_t a_row_stride, ptrdiff_t a_col_stride,
    ptrdiff_t b_stride, ptrdiff_t b_row_stride, ptrdiff_t b_col_stride,
    size_t block_size,
    cl_context context,
    cl_device_id device,
    cl_command_queue cl_queue,
    cl_program program) {

    cl_int clerr;
    cl_kernel kernel = clCreateKernel(program, "general_gemm", &clerr);
    if (clerr != CL_SUCCESS || kernel == nullptr) {
        std::cout << clErrorString(clerr) << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    int arg_idx = 0;
    void *a_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&a_svm, ((batch - 1) * a_stride + (m - 1) * a_row_stride + (k - 1) * a_col_stride + 1) * dtypeSize(dtype));
        infinirtMemcpy(a_svm, a, ((batch - 1) * a_stride + (m - 1) * a_row_stride + (k - 1) * a_col_stride + 1) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, a_svm);
    }
    void *b_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&b_svm, ((batch - 1) * b_stride + (k - 1) * b_row_stride + (n - 1) * b_col_stride + 1) * dtypeSize(dtype));
        infinirtMemcpy(b_svm, b, ((batch - 1) * b_stride + (k - 1) * b_row_stride + (n - 1) * b_col_stride + 1) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, b_svm);
    }
    void *c_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, c);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&c_svm, ((batch - 1) * c_stride + (m - 1) * c_row_stride + (n - 1) * c_col_stride + 1) * dtypeSize(dtype));
        infinirtMemcpy(c_svm, c, ((batch - 1) * c_stride + (m - 1) * c_row_stride + (n - 1) * c_col_stride + 1) * dtypeSize(dtype), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel, arg_idx++, c_svm);
    }
    cl_int a_s = static_cast<cl_int>(a_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &a_s);
    cl_int a_r_s = static_cast<cl_int>(a_row_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &a_r_s);
    cl_int a_c_s = static_cast<cl_int>(a_col_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &a_c_s);
    cl_int b_s = static_cast<cl_int>(b_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &b_s);
    cl_int b_r_s = static_cast<cl_int>(b_row_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &b_r_s);
    cl_int b_c_s = static_cast<cl_int>(b_col_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &b_c_s);
    cl_int c_s = static_cast<cl_int>(c_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &c_s);
    cl_int c_r_s = static_cast<cl_int>(c_row_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &c_r_s);
    cl_int c_c_s = static_cast<cl_int>(c_col_stride);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &c_c_s);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &batch);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &m);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &n);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_int), &k);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &alpha);
    clerr |= clSetKernelArg(kernel, arg_idx++, sizeof(float), &beta);

    size_t global_size[2] = {m * n, batch};
    size_t local_size[2] = {block_size, 1};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    if (c_svm) { // for python test
        infinirtMemcpy(c, c_svm, ((batch - 1) * c_stride + (m - 1) * c_row_stride + (n - 1) * c_col_stride + 1) * dtypeSize(dtype), INFINIRT_MEMCPY_D2H);
    }

    clReleaseKernel(kernel);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    if (_info.is_transed) {
        std::swap(a, b);
    }
    // _dtype
    auto batch = _info.batch;
    auto m = _info.m;
    auto n = _info.n;
    auto k = _info.k;

    auto c_stride = _info.c_matrix.stride;
    auto c_row_stride = _info.c_matrix.row_stride;
    auto c_col_stride = _info.c_matrix.col_stride;
    auto a_stride = _info.a_matrix.stride;
    auto a_row_stride = _info.a_matrix.row_stride;
    auto a_col_stride = _info.a_matrix.col_stride;
    auto b_stride = _info.b_matrix.stride;
    auto b_row_stride = _info.b_matrix.row_stride;
    auto b_col_stride = _info.b_matrix.col_stride;

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
    if (_dtype == INFINI_DTYPE_F16) {
        dt_val = "half";
    } else if (_dtype == INFINI_DTYPE_F32) {
        dt_val = "float";
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Tval=" + dt_val + " ";
    if (_dtype == INFINI_DTYPE_F16) {
        build_opts += "-D USE_HALF=1 ";
    }
    build_opts += "-cl-std=CL2.0 ";
    auto prog_shared = this->_opaque->internal->programCache()->getOrBuildWithSource("gemm", GemmKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(c, a, b, _dtype, batch, m, n, k, alpha, beta,
                              c_stride, c_row_stride, c_col_stride, a_stride, a_row_stride, a_col_stride, b_stride, b_row_stride, b_col_stride, block_size, clcontext, cldevice, clqueue, clprogram));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::opencl
