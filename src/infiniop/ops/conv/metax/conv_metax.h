#ifndef __CONV_METAX_H__
#define __CONV_METAX_H__

#include "../conv.h"

DESCRIPTOR(metax)

namespace op::conv::metax {
infiniStatus_t launchDirectConv2d(
    infiniDtype_t dtype,
    const ConvInfo &info,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream);

infiniStatus_t launchPatchEmbedConv3d(
    infiniDtype_t dtype,
    const ConvInfo &info,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream);
} // namespace op::conv::metax

#endif // __CONV_METAX_H__
