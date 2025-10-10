#pragma once

#include "../utils/infini_status_string.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

inline struct SpdlogInitializer {
    SpdlogInitializer() {
        if (!std::getenv("INFINICORE_LOG_LEVEL")) {
            spdlog::set_level(spdlog::level::off);
        } else {
            spdlog::cfg::load_env_levels("INFINICORE_LOG_LEVEL");
        }
    }
} spdlog_initializer;

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define INFINICORE_CHECK_ERROR(call)                                                                         \
    do {                                                                                                     \
        spdlog::info("Entering `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");                     \
        infiniStatus_t ret = (call);                                                                         \
        spdlog::info("Exiting `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");                      \
        if (ret != INFINI_STATUS_SUCCESS) {                                                                  \
            throw std::runtime_error(#call " failed with error: " + std::string(infini_status_string(ret))); \
        }                                                                                                    \
    } while (false)
