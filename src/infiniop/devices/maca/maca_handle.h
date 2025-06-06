#ifndef __INFINIOP_MACA_HANDLE_H__
#define __INFINIOP_MACA_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device::maca {
struct Handle : public InfiniopHandle {
    Handle(int device_id);
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);

protected:
    Handle(infiniDevice_t device, int device_id);

private:
    std::shared_ptr<Internal> _internal;
};

} // namespace device::maca

#endif // __INFINIOP_MACA_HANDLE_H__
