#ifndef __INFINIOP_MARS_HANDLE_H__
#define __INFINIOP_MARS_HANDLE_H__

#include "../metax/metax_handle.h"

namespace device::mars {

struct Handle final : public device::metax::Handle {
    explicit Handle(int device_id)
        : device::metax::Handle(INFINI_DEVICE_MARS, device_id) {}

    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id) {
        *handle_ptr = new Handle(device_id);
        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace device::mars

#endif // __INFINIOP_MARS_HANDLE_H__
