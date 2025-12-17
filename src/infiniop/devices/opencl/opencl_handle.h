#ifndef __INFINIOP_OPENCL_HANDLE_H__
#define __INFINIOP_OPENCL_HANDLE_H__

#include "../../handle.h"
#include <memory>

namespace device {
namespace opencl {

struct Handle : public InfiniopHandle {
    Handle(int device_id);
    class Internal;
    auto internal() const -> const std::shared_ptr<Internal> &;

protected:
    Handle(infiniDevice_t device, int device_id);

public:
    static infiniStatus_t create(InfiniopHandle **handle_ptr, int device_id);

private:
    std::shared_ptr<Internal> _internal;
};

} // namespace opencl
} // namespace device

#endif // __INFINIOP_OPENCL_HANDLE_H__
