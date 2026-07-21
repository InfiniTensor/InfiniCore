#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>

namespace infinicore::debug_session {

inline const char *log_path() {
    const char *path = std::getenv("INFINI_DEBUG_SESSION_LOG");
    if (path != nullptr && path[0] != '\0') {
        return path;
    }
    return "/opt/offline/infinilm-metax-20260622/.cursor/debug-11084d.log";
}

inline void log(const char *hypothesis_id,
                const char *location,
                const char *message,
                const std::string &data_json,
                const char *run_id = "m2-classify") {
    std::ofstream f(log_path(), std::ios::app);
    if (!f) {
        return;
    }
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    f << "{\"sessionId\":\"11084d\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\""
      << hypothesis_id << "\",\"location\":\"" << location << "\",\"message\":\"" << message
      << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
}

} // namespace infinicore::debug_session
