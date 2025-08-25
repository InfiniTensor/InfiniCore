#ifndef __RMS_NORM_BACKWARD_METAX_H__
#define __RMS_NORM_BACKWARD_METAX_H__

#include "../rms_norm_backward.h"
#include "../info.h"

namespace op::rms_norm_backward::metax {

/**
 * @brief Template function for RMS Norm backward computation on Metax platform
 * 
 * @tparam BLOCK_SIZE Number of threads per block
 * @tparam Tdata Data type for input/output tensors
 * @tparam Tweight Weight tensor data type
 */
template<unsigned int BLOCK_SIZE, typename Tdata, typename Tweight>
infiniStatus_t rmsNormBackwardMetax(
    const RMSNormBackwardInfo &info,
    Tdata * grad_x,
    Tdata * grad_w,
    const Tdata * grad_y,
    const Tdata * x,
    const Tweight * w,
    void * stream,
    void * workspace
);

} // namespace op::rms_norm_backward::metax

DESCRIPTOR(metax)

#endif // __RMS_NORM_BACKWARD_METAX_H__