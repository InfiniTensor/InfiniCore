/**
 * Backward-compatible include for the InfLLM-v2 vendor shim.
 *
 * The InfLLM-v2 entrypoints are provided by an external shared library and are
 * now declared under `infinicore/adaptor/infllmv2_api.hpp` to make the
 * dependency boundary explicit.
 *
 * The vendor symbols themselves remain in the global namespace.
 */
#pragma once

#include "infinicore/adaptor/infllmv2_api.hpp"
