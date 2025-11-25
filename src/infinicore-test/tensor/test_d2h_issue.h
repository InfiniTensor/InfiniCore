#pragma once

#include "../test_runner.h"

namespace infinicore::test {

class D2HIssueTest : public TestFramework {
public:
    std::string getName() const override { return "D2HIssueTest"; }

    TestResult run() override;
};

} // namespace infinicore::test
