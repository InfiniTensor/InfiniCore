#ifndef __QUANTIZE_INFO_H__
#define __QUANTIZE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

// 需要一个值，要的是input的大小(x * y * z)
// 需要 out_q.dtype  out_s.stride
// input.size input.dim
//

namespace op::quantize {

class QuantizeInfo {
    QuantizeInfo() = default;

public:
    infiniopTensorDescriptor_t _input_desc, _output_q_desc, _output_s_desc;

    infiniopTensorDescriptor_t input() const { return _input_desc; }
    infiniopTensorDescriptor_t output_q() const { return _output_q_desc; }
    infiniopTensorDescriptor_t output_s() const { return _output_s_desc; }

    static utils::Result<QuantizeInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t output_q_desc,
           infiniopTensorDescriptor_t output_s_desc) {

        return utils::Result<QuantizeInfo>(QuantizeInfo{input_desc, output_q_desc, output_s_desc});
    }
};

} // namespace op::quantize

#endif // __DEQUANTIZE_INFO_H__