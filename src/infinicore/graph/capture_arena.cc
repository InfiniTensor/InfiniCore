#include "infinicore/graph/capture_arena.hpp"

#include "infinicore/context/context.hpp"
#include "../context/context_impl.hpp"
#include "../context/runtime/runtime.hpp"

#include <stdexcept>

#if defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#endif

namespace infinicore::graph {
namespace {

#if defined(ENABLE_ATEN)
DataType data_type_from_at(at::ScalarType st) {
    switch (st) {
    case at::kFloat:
        return DataType::F32;
    case at::kHalf:
        return DataType::F16;
    case at::kBFloat16:
        return DataType::BF16;
    case at::kInt:
        return DataType::I32;
    case at::kLong:
        return DataType::I64;
    case at::kByte:
        return DataType::U8;
    case at::kBool:
        return DataType::BOOL;
    default:
        throw std::runtime_error("CaptureArena: unsupported at::ScalarType");
    }
}
#endif

} // namespace

std::shared_ptr<Memory> CaptureArena::allocate(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }
    auto mem = context::allocateMemory(nbytes);
    bytes_allocated_ += mem ? mem->size() : 0;
    memories_.push_back(mem);
    return mem;
}

Tensor CaptureArena::empty(const Shape &shape, DataType dtype, Device device) {
    Tensor t = Tensor::empty(shape, dtype, device);
    // Keep Tensor handle so Memory stays live for torch aliases with no-op deleters.
    bytes_allocated_ += t->nbytes();
    tensors_.push_back(t);
    return t;
}

#if defined(ENABLE_ATEN)
at::Tensor CaptureArena::empty_aten(at::IntArrayRef sizes, at::TensorOptions options) {
    Shape shape(sizes.begin(), sizes.end());
    const DataType dtype = data_type_from_at(options.dtype().toScalarType());
    Device device = context::getDevice();
    if (options.device().is_cuda()) {
        device = Device(device.getType(), static_cast<Device::Index>(options.device().index()));
    }
    Tensor ic = empty(shape, dtype, device);
    return adaptor::to_aten_tensor(ic);
}

void CaptureArena::retain(at::Tensor t) {
    if (t.defined()) {
        torch_retain_.push_back(std::move(t));
    }
}
#endif

CaptureArena *current_capture_arena() {
    // Use context TLS (same TU as isDeviceStreamCapturing) to avoid duplicate
    // TLS across static archives / multiple DSO copies.
    return context::currentCaptureArena();
}

void begin_capture_arena(CaptureArena &arena) {
    if (context::currentCaptureArena() != nullptr) {
        throw std::runtime_error("begin_capture_arena: nested capture arena not supported");
    }
    arena.set_active(true);
    context::setCurrentCaptureArena(&arena);
    // Freeze new IC allocations for segment lifetime (same pin contract as graph record).
    ContextImpl::singleton().getCurrentRuntime()->setDeviceAllocatorPinMode(true);
}

void end_capture_arena(CaptureArena &arena) {
    if (context::currentCaptureArena() != &arena) {
        throw std::runtime_error("end_capture_arena: arena mismatch");
    }
    arena.set_active(false);
    context::setCurrentCaptureArena(nullptr);
    ContextImpl::singleton().getCurrentRuntime()->setDeviceAllocatorPinMode(false);
}

bool capture_arena_active() {
    auto *a = context::currentCaptureArena();
    return a != nullptr && a->active();
}

size_t capture_arena_bytes_current() {
    auto *a = context::currentCaptureArena();
    return a != nullptr ? a->bytes_allocated() : 0;
}

size_t capture_arena_blocks_current() {
    auto *a = context::currentCaptureArena();
    return a != nullptr ? a->num_blocks() : 0;
}

bool capture_used_torch_mempool() {
    return false;
}

} // namespace infinicore::graph
