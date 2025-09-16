#ifndef __INFINICORE_DEVICE_API_HPP__
#define __INFINICORE_DEVICE_API_HPP__

#include <cstdint>
#include <string>

namespace infinicore {

class Device {
public:
    using Index = std::size_t;

    enum class Type {
        cpu,
        cuda,
        meta,
    };

    Device(const Type &type, const Index &index = 0);

    const Type &get_type() const;

    const Index &get_index() const;

    std::string to_string() const;

    static std::string to_string(const Type &type);

private:
    Type type_;

    Index index_;
};

} // namespace infinicore

#endif
