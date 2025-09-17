#ifndef __INFINICORE_DEVICE_API_HPP__
#define __INFINICORE_DEVICE_API_HPP__

#include <cstdint>
#include <string>

namespace infinicore {

class Device {
public:
    using Index = std::size_t;

    enum class Type {
        CPU,
        CUDA,
        META,
    };

    Device(const Type &type, const Index &index = 0);

    const Type &getType() const;

    const Index &getIndex() const;

    std::string toString() const;

    static std::string toString(const Type &type);

private:
    Type type_;

    Index index_;
};

} // namespace infinicore

#endif
