#pragma once

#include "../device.hpp"
#include "../dtype.hpp"
#include "../memory.hpp"
#include "../tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(ENABLE_ATEN)
#include <ATen/ATen.h>
#endif

namespace infinicore::graph {

/// Owns capture-lifetime device memory for one hcGraph device segment.
///
/// During ``hcStreamBeginCapture``…``EndCapture``, MoE / ATen temps allocate
/// (or are retained) here so replay addresses stay valid without
/// ``c10::cuda::MemPool``. Torch tensors are aliases (``from_blob`` / no-op
/// deleter) on InfiniCore-owned pointers when allocated via ``empty`` /
/// ``empty_aten``.
class CaptureArena {
public:
    CaptureArena() = default;
    ~CaptureArena() = default;

    CaptureArena(const CaptureArena &) = delete;
    CaptureArena &operator=(const CaptureArena &) = delete;

    /// Allocate via InfiniCore device allocator; retain until arena destruction.
    std::shared_ptr<Memory> allocate(size_t nbytes);

    /// Contiguous empty tensor on device; storage retained by this arena.
    Tensor empty(const Shape &shape, DataType dtype, Device device);

#if defined(ENABLE_ATEN)
    /// IC-backed ``at::Tensor`` alias (no-op deleter); preferred MoE workspace path.
    at::Tensor empty_aten(at::IntArrayRef sizes, at::TensorOptions options);

    /// Retain a torch-allocator tensor for segment lifetime (bridge for ops that
    /// still allocate via CUDACachingAllocator under capture, e.g. aten routing).
    void retain(at::Tensor t);
#endif

    size_t bytes_allocated() const { return bytes_allocated_; }
    size_t num_blocks() const { return memories_.size() + tensors_.size(); }
    size_t num_retained_torch() const {
#if defined(ENABLE_ATEN)
        return torch_retain_.size();
#else
        return 0;
#endif
    }

    bool active() const { return active_; }
    void set_active(bool v) { active_ = v; }

private:
    bool active_{false};
    size_t bytes_allocated_{0};
    std::vector<std::shared_ptr<Memory>> memories_;
    std::vector<Tensor> tensors_;
#if defined(ENABLE_ATEN)
    std::vector<at::Tensor> torch_retain_;
#endif
};

CaptureArena *current_capture_arena();

/// Activate ``arena`` for the current thread (pin InfiniCore allocator).
void begin_capture_arena(CaptureArena &arena);

/// Deactivate TLS; ``arena`` remains owned by ``DeviceGraph`` for replay.
void end_capture_arena(CaptureArena &arena);

/// True while a capture arena is active on this thread.
bool capture_arena_active();

size_t capture_arena_bytes_current();
size_t capture_arena_blocks_current();

/// Always false after Phase 3 (MemPool path removed). Smoke asserts this.
bool capture_used_torch_mempool();

} // namespace infinicore::graph
