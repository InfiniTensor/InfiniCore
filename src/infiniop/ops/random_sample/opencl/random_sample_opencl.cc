#include "random_sample_opencl.h"
#include "../../../../infinirt/opencl/infinirt_opencl.h"
#include "../../../devices/opencl/opencl_common.h"
#include "../info.h"
#include <fstream>
#include <math.h>
#include <memory>
#include <sstream>

static const char *RandomSampleKernelSource = R"CLC(
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef Tval
#define Tval float
#endif
#ifndef Tidx
#define Tidx unsigned int
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 1024
#endif

typedef unsigned int T_idx;

typedef struct {
    Tidx idx;
    Tval val;
} KVPair;

KVPair group_argmax(local KVPair *data, KVPair reg) {
    T_idx const idx = get_local_id(0),
               len = get_local_size(0);

    data[idx] = reg;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (T_idx stride = len >> 1; stride; stride >>= 1) {
        if (idx < stride) {
            local KVPair
                *a = data + idx,
                *b = data + idx + stride;
            if (b->val > a->val) *a = *b;
            else if (b->val == a->val) {
                if(b->idx < a->idx) *a = *b;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return data[0];
}

kernel void argmax_build_pairs(
    global Tval const *input,
    global KVPair *output,
    T_idx const n,
    float init) {

    T_idx const
        g_idx = get_global_id(0),
        g_len = get_global_size(0),
        l_idx = get_local_id(0);

    KVPair reg = {-1, (Tval) init};
    for (T_idx i = g_idx; i < n; i += g_len) {
        Tval const val = input[i];
        if (val > reg.val) reg = (KVPair) {i, val};
    }

    local KVPair kv_pairs[GROUP_SIZE];
    reg = group_argmax(kv_pairs, reg);

    if (l_idx == 0) output[g_idx / GROUP_SIZE] = reg;
}

kernel void argmax_reduce(
    global KVPair const *pairs,
    global Tidx *output,
    T_idx const n,
    float init) {

    T_idx const
        g_idx = get_global_id(0),
        g_len = get_global_size(0),
        l_idx = get_local_id(0);

    KVPair reg = {-1, (Tval) init};
    for (T_idx i = g_idx; i < n; i += g_len) {
        KVPair const pair = pairs[i];
        if (pair.val > reg.val) reg = pair;
        else if (pair.val == reg.val) {
            if (pair.idx < reg.idx) reg = pair;
        }
    }

    local KVPair kv_pairs[GROUP_SIZE];
    reg = group_argmax(kv_pairs, reg);

    // 最终结果写回 global
    if (l_idx == 0) *output = reg.idx;
}


)CLC";

static size_t alignTo(size_t x, size_t a) { return (x + a - 1) / a * a; }

namespace op::random_sample::opencl {

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
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::opencl::Handle *>(handle_);
    auto internal = handle->internal();

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    size_t workspace_size;
    size_t wg = internal->maxThreadsPerBlock();
    size_t num_partials = (info.n + wg - 1) / wg;
    size_t argmax_tmp = num_partials * (dtypeSize(info.dt_i) + dtypeSize(info.dt_p));
    workspace_size = alignTo(argmax_tmp + 256, 256);

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

infiniStatus_t launchKernel(
    const void *probs, size_t n, void *result, void *workspace, size_t workspace_size,
    infiniDtype_t dt_i, infiniDtype_t dt_p, float random_val, float topp, int topk,
    float temperature, size_t block_size, cl_context context, cl_device_id device,
    cl_command_queue cl_queue, cl_program program) {
    // todo: add random
    // argmax
    if (topk != 1) {
        std::cout << " only argmax" << std::endl;
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    cl_int clerr;
    size_t n_pairs = (n + block_size - 1) / block_size / 2;
    size_t reduce_size = std::min(n_pairs, block_size);

    cl_kernel kernel_0 = clCreateKernel(program, "argmax_build_pairs", &clerr);
    if (clerr != CL_SUCCESS || kernel_0 == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_kernel kernel_1 = clCreateKernel(program, "argmax_reduce", &clerr);
    if (clerr != CL_SUCCESS || kernel_1 == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    int arg_idx = 0;
    void *probs_svm = NULL;
    clerr = clSetKernelArgSVMPointer(kernel_0, arg_idx++, probs);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&probs_svm, n * dtypeSize(dt_p));
        infinirtMemcpy(probs_svm, probs, n * dtypeSize(dt_p), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel_0, arg_idx++, probs_svm);
    }

    void *w_svm = NULL;
    clerr |= clSetKernelArgSVMPointer(kernel_0, arg_idx++, workspace);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&w_svm, workspace_size);
        infinirtMemcpy(w_svm, workspace, workspace_size, INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel_0, arg_idx++, w_svm);
    }
    cl_int len = static_cast<cl_int>(n);
    clerr |= clSetKernelArg(kernel_0, arg_idx++, sizeof(cl_int), &len);

    float neg_inf = -INFINITY;
    clerr |= clSetKernelArg(kernel_0, arg_idx++, sizeof(float), &neg_inf);

    size_t global_size[1] = {n_pairs * block_size};
    size_t local_size[1] = {block_size};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel_0, 1, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel_0);
        clReleaseKernel(kernel_1);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    arg_idx = 0;
    if (!w_svm) {
        clerr |= clSetKernelArgSVMPointer(kernel_1, arg_idx++, workspace);
    } else {
        clerr = clSetKernelArgSVMPointer(kernel_1, arg_idx++, w_svm);
    }
    void *result_svm = NULL;
    clerr |= clSetKernelArgSVMPointer(kernel_1, arg_idx++, result);
    if (clerr != CL_SUCCESS) { // for python test
        infinirtMalloc(&result_svm, sizeof(dt_i));
        infinirtMemcpy(result_svm, result, sizeof(dt_i), INFINIRT_MEMCPY_H2D);
        arg_idx -= 1;
        clerr = clSetKernelArgSVMPointer(kernel_1, arg_idx++, result_svm);
    }

    len = static_cast<cl_int>(n_pairs);
    clerr |= clSetKernelArg(kernel_1, arg_idx++, sizeof(cl_int), &len);
    clerr |= clSetKernelArg(kernel_1, arg_idx++, sizeof(float), &neg_inf);

    global_size[0] = {reduce_size};
    local_size[0] = {reduce_size};

    clerr = clEnqueueNDRangeKernel(cl_queue, kernel_1, 1, nullptr, global_size, local_size, 0, nullptr, nullptr);
    if (clerr != CL_SUCCESS) {
        fprintf(stderr, "[OpenCL] clEnqueueNDRangeKernel failed: %s (%d)\n", clErrorString(clerr), clerr);
        fprintf(stderr, "  global_size: %zu, local_size: %zu\n", global_size[0], local_size[0]);
        clReleaseKernel(kernel_0);
        clReleaseKernel(kernel_1);
        clReleaseProgram(program);
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    if (result_svm) { // for python test
        infinirtMemcpy(result, result_svm, dtypeSize(dt_i), INFINIRT_MEMCPY_D2H);
    }

    // cleanup kernel
    clReleaseKernel(kernel_0);
    clReleaseKernel(kernel_1);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
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

    size_t block_size = _opaque->internal->maxThreadsPerBlock();
    auto dt_i = _info.dt_i;
    auto dt_p = _info.dt_p;
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

    // Build program with cache
    std::string dt_probs, dt_idx;
    if (!dtypeToClType(dt_p, dt_probs)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!dtypeToClType(dt_i, dt_idx)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // build options
    std::string build_opts;
    build_opts += "-D Tval=" + dt_probs + " ";
    build_opts += "-D Tidx=" + dt_idx + " ";
    build_opts += "-D GROUP_SIZE=" + std::to_string(block_size) + " ";
    if (dt_p == INFINI_DTYPE_F16) {
        build_opts += "-D USE_HALF=1 ";
    }
    build_opts += "-cl-std=CL2.0 ";

    auto prog_shared = _opaque->internal->programCache()->getOrBuildWithSource("random_sample", RandomSampleKernelSource, build_opts, clcontext, cldevice);
    if (!prog_shared) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_program clprogram = reinterpret_cast<cl_program>(prog_shared.get());

    CHECK_STATUS(launchKernel(probs, _info.n, result, workspace, workspace_size, dt_i, dt_p, random_val, topp, topk,
                              temperature, block_size, clcontext, cldevice, clqueue, clprogram));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::opencl
