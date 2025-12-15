#include "logical_xor_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <mutex>
#include <unordered_map>

namespace op::logical_xor::cpu {

// Track input dtype for each Descriptor so we can support both
//   - out-of-place / inplace(out) with bool output
//   - inplace(a) / inplace(b) with output dtype == input dtype (int32/uint8)
static std::unordered_map<const Descriptor *, infiniDtype_t> g_input_dtype;
static std::mutex g_input_dtype_mutex;

Descriptor::~Descriptor() {
    std::lock_guard<std::mutex> lock(g_input_dtype_mutex);
    g_input_dtype.erase(this);
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // True output dtype (memory layout of output buffer)
    auto out_dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // Input dtype(s)
    auto a_dtype = a_desc->dtype();
    auto b_dtype = b_desc->dtype();

    // Inputs must have the same dtype and be one of the supported ones
    CHECK_OR_RETURN(a_dtype == b_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_DTYPE(a_dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I32, INFINI_DTYPE_U8);

    // Output must be either bool (standard case) or equal to input dtype
    if (!(out_dtype == INFINI_DTYPE_BOOL || out_dtype == a_dtype)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // Here we pass the *output* dtype into the descriptor, keeping the
    // semantics of CREATE_ELEMENTWISE_CPU_DESCRIPTOR consistent with
    // other elementwise ops.
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, out_dtype, out_desc, input_desc_vec);

    // Remember the common input dtype for this descriptor
    {
        std::lock_guard<std::mutex> lock(g_input_dtype_mutex);
        g_input_dtype[*desc_ptr] = a_dtype;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // _dtype now represents the output dtype (as in other elementwise ops)
    auto out_dtype = _dtype;

    // Look up the input dtype for this descriptor
    infiniDtype_t in_dtype;
    {
        std::lock_guard<std::mutex> lock(g_input_dtype_mutex);
        auto it = g_input_dtype.find(this);
        if (it == g_input_dtype.end()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        in_dtype = it->second;
    }

    // Case 1: boolean inputs (and thus boolean output)
    if (in_dtype == INFINI_DTYPE_BOOL) {
        // All bool: we can use the homogeneous path
        return _device_info->calculate<LogicalXorOp, bool>(_info, output, inputs, stream);
    }

    // Case 2: integer inputs (int32 / uint8)
    if (in_dtype == INFINI_DTYPE_I32) {
        if (out_dtype == INFINI_DTYPE_BOOL) {
            // Inputs int32, output bool
            return _device_info->calculate<LogicalXorOp, bool, int32_t, int32_t>(_info, output, inputs, stream);
        } else if (out_dtype == INFINI_DTYPE_I32) {
            // Inplace(a/b): inputs and output are int32
            // Use homogeneous path; LogicalXorOp returns bool which is
            // implicitly converted to int32 (0/1).
            return _device_info->calculate<LogicalXorOp, int32_t>(_info, output, inputs, stream);
        }
    } else if (in_dtype == INFINI_DTYPE_U8) {
        if (out_dtype == INFINI_DTYPE_BOOL) {
            // Inputs uint8, output bool
            return _device_info->calculate<LogicalXorOp, bool, uint8_t, uint8_t>(_info, output, inputs, stream);
        } else if (out_dtype == INFINI_DTYPE_U8) {
            // Inplace(a/b): inputs and output are uint8
            return _device_info->calculate<LogicalXorOp, uint8_t>(_info, output, inputs, stream);
        }
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}
} // namespace op::logical_xor::cpu

