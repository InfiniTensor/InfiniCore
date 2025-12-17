#ifndef __INFINIOP_OPENCL_COMMON_H__
#define __INFINIOP_OPENCL_COMMON_H__

#include "../../../utils.h"
#include "../pool.h"
#include "opencl_handle.h"
#include "opencl_kernel_common.h"
#include "opencl_program_cache.h"
#include <functional>
#include <vector>

namespace device::opencl {

class Handle::Internal {

    int _warp_size,
        _max_threads_per_block,
        _block_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    ProgramCache *programCache() const;

private:
    std::unique_ptr<ProgramCache> program_cache_;
};

} // namespace device::opencl

#endif // __INFINIOP_OPENCL_COMMON_H__
