#include "all_equal_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cstddef>

namespace op::all_equal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    auto dtype = a_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(a_shape, b_shape);

    auto info_result = AllEqualInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(nullptr, info_result.take(), 0, handle_->device, handle_->device_id);

    return INFINI_STATUS_SUCCESS;
}

// Perform elementwise operation when all inputs have the same type
template <typename Op, typename Tout, typename Tin>
void calculate_impl(const op::all_equal::AllEqualInfo &info,
                    void *output,
                    const std::vector<const void *> &inputs) {
    Tout *out = reinterpret_cast<Tout *>(output);
    auto input_a = reinterpret_cast<const Tin *>(inputs[0]);
    auto input_b = reinterpret_cast<const Tin *>(inputs[1]);
    const ptrdiff_t input_numel = static_cast<ptrdiff_t>(info.getInputNumel());

    bool all_equal = true;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < input_numel; ++i) {
        auto get_input_idx = [&](size_t input_id) {
            return info.getInputContiguous()[input_id]
                     ? i
                     : (info.getInputBroadcasted()[input_id]
                            ? op::common_cpu::indexToReducedOffset(i, info.getNdim(), info.getDefaultStrides(), info.getInputStrides(input_id))
                            : op::common_cpu::indexToOffset(i, info.getNdim(), info.getInputShape(input_id), info.getInputStrides(input_id)));
        };
        if constexpr (std::is_same_v<Tin, fp16_t> || std::is_same_v<Tin, bf16_t>) {
            Tout elem = Op{}.template operator()<Tout, float>(utils::cast<float>(input_a[get_input_idx(0)]), utils::cast<float>(input_b[get_input_idx(1)]));
            if (elem == false) {
                all_equal = false;
            }
        } else {
            Tout elem = Op{}.template operator()<Tout, Tin>(input_a[(get_input_idx(0))], input_b[get_input_idx(1)]);
            if (elem == false) {
                all_equal = false;
            }
        }
    }
    *out = all_equal ? true : false;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_info._dtype) {
    case INFINI_DTYPE_BOOL:
        calculate_impl<AllEqualOp, bool, bool>(_info, output, inputs);
        break;
    case INFINI_DTYPE_I8:
        calculate_impl<AllEqualOp, bool, int8_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_I16:
        calculate_impl<AllEqualOp, bool, int16_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_I32:
        calculate_impl<AllEqualOp, bool, int32_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_I64:
        calculate_impl<AllEqualOp, bool, int64_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_BF16:
        calculate_impl<AllEqualOp, bool, bf16_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_F16:
        calculate_impl<AllEqualOp, bool, fp16_t>(_info, output, inputs);
        break;
    case INFINI_DTYPE_F32:
        calculate_impl<AllEqualOp, bool, float>(_info, output, inputs);
        break;
    case INFINI_DTYPE_F64:
        calculate_impl<AllEqualOp, bool, double>(_info, output, inputs);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::all_equal::cpu
