#pragma once

#include "infinicore/device.hpp"
#include <infinirt.h>

namespace infinicore::graph::standalone_infinirt {

bool enabled();

bool available(const Device &device);

infiniStatus_t set_device(const Device &device);

infiniStatus_t stream_begin_capture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode);

infiniStatus_t stream_end_capture(infinirtStream_t stream, infinirtGraph_t *graph);

infiniStatus_t graph_destroy(infinirtGraph_t graph);

infiniStatus_t graph_instantiate(infinirtGraphExec_t *graph_exec, infinirtGraph_t graph);

infiniStatus_t graph_exec_destroy(infinirtGraphExec_t graph_exec);

infiniStatus_t graph_launch(infinirtGraphExec_t graph_exec, infinirtStream_t stream);

} // namespace infinicore::graph::standalone_infinirt
