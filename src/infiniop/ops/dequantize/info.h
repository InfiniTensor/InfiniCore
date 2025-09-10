#ifndef __DEQUANTIZE_INFO_H__
#define __DEQUANTIZE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::dequantize {

class DequantizeInfo {
    DequantizeInfo() = default;

public:
    int _in_c, _qout_c, _G;

    int in_c() const { return _in_c; }
    int qout_c() const { return _qout_c; }
    int G() const { return _G; }

    static utils::Result<DequantizeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t qweight_desc,
        infiniopTensorDescriptor_t scales_desc,
        infiniopTensorDescriptor_t zeros_desc) {

        int _in_c = static_cast<int>(qweight_desc->dim(0));
        int _qout_c = static_cast<int>(qweight_desc->dim(1));
        int _G = static_cast<int>(scales_desc->dim(0));

        return utils::Result<DequantizeInfo>(DequantizeInfo{
            _in_c,
            _qout_c,
            _G});
    }
};

} // namespace op::dequantize

#endif // __DEQUANTIZE_INFO_H__
