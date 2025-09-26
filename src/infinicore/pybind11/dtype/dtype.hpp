#pragma once
#include "infinicore.hpp"

namespace infinicore::py {

class DataType {
public:
    DataType(const infinicore::DataType &dtype);

    operator infinicore::DataType() const {
        return dtype_;
    }

    static std::string toString(const DataType &dtype);

private:
    infinicore::DataType dtype_;
};

} // namespace infinicore::py
