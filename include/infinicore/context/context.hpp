#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include "../graph/graph.hpp"

#include <infiniop.h>
#include <infinirt.h>

#include <cstdint>
#include <memory>

namespace infinicore {

namespace graph {
class CaptureArena;
}

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);

infinirtStream_t getStream();
infiniopHandle_t getInfiniopHandle(Device device);

void syncStream();
void syncDevice();

/// Sync then return idle InfiniCore device allocator blocks to the driver.
void trimDeviceMemory();

/// Free pinned-host allocations deferred while another device was active.
void flushDeferredPinnedHostFrees();

std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyH2H(void *dst, const void *src, size_t size);

// Timing APIs for performance measurement
infinirtEvent_t createEvent();
infinirtEvent_t createEventWithFlags(uint32_t flags);
void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
bool queryEvent(infinirtEvent_t event);
void synchronizeEvent(infinirtEvent_t event);
void destroyEvent(infinirtEvent_t event);
float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);
void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Graph recording APIs
bool isGraphRecording();
void startGraphRecording();
void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
std::shared_ptr<graph::Graph> stopGraphRecording();

/// True while ``Graph::instantiate`` is inside ``hcStreamBeginCapture``…
/// ``EndCapture``. Host-break ops (e.g. Triton MoE) must not run on this path.
bool isDeviceStreamCapturing();
void setDeviceStreamCapturing(bool capturing);

/// Active CaptureArena for the current thread (set around stream capture).
graph::CaptureArena *currentCaptureArena();
void setCurrentCaptureArena(graph::CaptureArena *arena);

/// Engine TLS inference phase for phase-scoped FA / MoE capture policy.
enum class InferencePhase : uint8_t {
    Unknown = 0,
    Prefill = 1,
    Decode = 2,
};

void setInferencePhase(InferencePhase phase);
InferencePhase getInferencePhase();

/// ``INFINI_CUDAGRAPH_POLICY``: empty (legacy), ``eager``, or ``full_and_piecewise``.
/// Unknown values (including ``track_b``) are treated as empty / legacy.
const char *cudagraphPolicy();

/// FA may enter a device segment: diagnose ``INFINI_FA_FORCE_CAPTURE=1``, else
/// ``full_and_piecewise`` ∧ decode phase only. Default / eager → host-break.
bool faInGraphAllowed();

/// Triton MoE under stream capture: explicit ``INFINI_MOE_TRITON_CAPTURE``, else
/// ``full_and_piecewise`` ∧ decode. Explicit ``eager`` / unset policy without
/// the legacy env → false.
bool moeTritonCaptureAllowed();

/// RAII restore of ``InferencePhase`` for graph capture / forward scopes.
class InferencePhaseGuard {
public:
    explicit InferencePhaseGuard(InferencePhase phase)
        : prev_(getInferencePhase()) {
        setInferencePhase(phase);
    }
    ~InferencePhaseGuard() { setInferencePhase(prev_); }
    InferencePhaseGuard(const InferencePhaseGuard &) = delete;
    InferencePhaseGuard &operator=(const InferencePhaseGuard &) = delete;

private:
    InferencePhase prev_;
};

} // namespace context

} // namespace infinicore
