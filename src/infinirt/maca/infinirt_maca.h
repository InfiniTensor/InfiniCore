#ifndef __INFINIRT_MACA_H__
#define __INFINIRT_MACA_H__
#include "../infinirt_impl.h"

namespace infinirt::maca {
#ifdef ENABLE_METAX_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::maca

#endif // __INFINIRT_MACA_H__
