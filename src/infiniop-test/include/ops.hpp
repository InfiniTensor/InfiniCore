#ifndef __INFINIOPTEST_OPS_HPP__
#define __INFINIOPTEST_OPS_HPP__
#include "test.hpp"

/*
 * Declare all the tests here
 */
DECLARE_INFINIOP_TEST(gemm)
DECLARE_INFINIOP_TEST(random_sample)
DECLARE_INFINIOP_TEST(rms_norm)
DECLARE_INFINIOP_TEST(mul)
DECLARE_INFINIOP_TEST(rope)
DECLARE_INFINIOP_TEST(clip)
DECLARE_INFINIOP_TEST(swiglu)
DECLARE_INFINIOP_TEST(add)
DECLARE_INFINIOP_TEST(causal_softmax)
DECLARE_INFINIOP_TEST(rearrange)
DECLARE_INFINIOP_TEST(sub)
DECLARE_INFINIOP_TEST(div)
DECLARE_INFINIOP_TEST(silu)
DECLARE_INFINIOP_TEST(logical_and)
DECLARE_INFINIOP_TEST(logical_or)
DECLARE_INFINIOP_TEST(equal)
DECLARE_INFINIOP_TEST(relu_backward)
DECLARE_INFINIOP_TEST(gelu)
DECLARE_INFINIOP_TEST(gelu_backward)
DECLARE_INFINIOP_TEST(cross_entropy_loss_backward)

#define REGISTER_INFINIOP_TEST(name)                      \
    {                                                     \
        #name,                                            \
        {                                                 \
            infiniop_test::name::Test::build,             \
            infiniop_test::name::Test::attribute_names(), \
            infiniop_test::name::Test::tensor_names(),    \
            infiniop_test::name::Test::output_names(),    \
        }},

/*
 * Register all the tests here
 */
#define TEST_BUILDER_MAPPINGS                               \
    {                                                       \
        REGISTER_INFINIOP_TEST(gemm)                        \
        REGISTER_INFINIOP_TEST(random_sample)               \
        REGISTER_INFINIOP_TEST(add)                         \
        REGISTER_INFINIOP_TEST(mul)                         \
        REGISTER_INFINIOP_TEST(clip)                        \
        REGISTER_INFINIOP_TEST(swiglu)                      \
        REGISTER_INFINIOP_TEST(rope)                        \
        REGISTER_INFINIOP_TEST(rms_norm)                    \
        REGISTER_INFINIOP_TEST(causal_softmax)              \
        REGISTER_INFINIOP_TEST(rearrange)                   \
        REGISTER_INFINIOP_TEST(sub)                         \
        REGISTER_INFINIOP_TEST(div)                         \
        REGISTER_INFINIOP_TEST(silu)                        \
        REGISTER_INFINIOP_TEST(logical_and)                 \
        REGISTER_INFINIOP_TEST(logical_or)                  \
        REGISTER_INFINIOP_TEST(equal)                       \
        REGISTER_INFINIOP_TEST(relu_backward)               \
        REGISTER_INFINIOP_TEST(gelu)                        \
        REGISTER_INFINIOP_TEST(gelu_backward)               \
        REGISTER_INFINIOP_TEST(cross_entropy_loss_backward) \
    }

namespace infiniop_test {

// Global variable for {op_name: builder} mappings
extern std::unordered_map<std::string, const TestBuilder> TEST_BUILDERS;

template <typename V>
bool check_names(
    const std::unordered_map<std::string, V> &map,
    const std::vector<std::string> &names) {
    for (auto const &name : names) {
        if (map.find(name) == map.end()) {
            return false;
        }
    }
    return true;
}

} // namespace infiniop_test

#endif
