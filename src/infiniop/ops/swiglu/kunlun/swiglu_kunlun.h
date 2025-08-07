#ifndef __SWIGLU_KUNLUN_H__
#define __SWIGLU_KUNLUN_H__

#include "../../../elementwise/kunlun/elementwise_kunlun.h"

ELEMENTWISE_DESCRIPTOR(swiglu, kunlun)

// Op interface declare
LAUNCH_ELEMENTWISE_KERNEL(SwiGLU)

#endif // __SWIGLU_KUNLUN_H__
