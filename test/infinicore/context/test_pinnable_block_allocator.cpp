// Standalone regression test with a deterministic failing device allocator.
// Compile with pinnable_block_allocator.cc, device.cc, and the spdlog includes.
#include "../../../src/infinicore/context/allocators/pinnable_block_allocator.hpp"
#include <cassert>
#include <cstdlib>
#include <infinirt.h>
#include <iostream>
#include <unordered_set>

static std::unordered_set<void *> live;
static int fail_allocations = 0;
static int malloc_calls = 0;
extern "C" infiniStatus_t infinirtMalloc(void **ptr, size_t) {
    ++malloc_calls;
    if (fail_allocations > 0) {
        --fail_allocations;
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    *ptr = std::malloc(1);
    live.insert(*ptr);
    return INFINI_STATUS_SUCCESS;
}
extern "C" infiniStatus_t infinirtFree(void *ptr) {
    assert(live.erase(ptr) == 1);
    std::free(ptr);
    return INFINI_STATUS_SUCCESS;
}
int main() {
    using infinicore::PinnableBlockAllocator;
    {
        PinnableBlockAllocator alloc(infinicore::Device::cpu());
        auto *cached = alloc.allocate(1);
        alloc.deallocate(cached);
        alloc.set_pin_mode(true);
        auto *graph = alloc.allocate(1);
        assert(graph == cached);
        alloc.set_pin_mode(false);
        alloc.deallocate(graph);
        alloc.trim();
        assert(live.count(graph) == 1); // Reused graph memory must stay frozen.
        alloc.mark_in_use_(graph, true);
        alloc.deallocate(graph);

        auto *active = alloc.allocate(40000);
        alloc.deallocate(active);
        alloc.mark_in_use_(active, true); // A graph reactivates a free-list entry.
        alloc.trim();
        assert(live.count(active) == 1);
        alloc.deallocate(active);
        auto *unused = alloc.allocate(40000);
        alloc.deallocate(unused);
        auto before = malloc_calls;
        fail_allocations = 1;
        auto *retried = alloc.allocate(300000);
        assert(malloc_calls == before + 2);
        assert(live.count(graph) == 1);
        assert(live.size() == 2); // Only frozen graph and new allocation remain.
        alloc.deallocate(retried);
        alloc.trim();
        assert(live.size() == 1);

        alloc.set_pin_mode(true);
        before = malloc_calls;
        fail_allocations = 1;
        bool threw = false;
        try {
            alloc.allocate(300000);
        } catch (const std::runtime_error &) { threw = true; }
        assert(threw && malloc_calls == before + 1);
        assert(live.count(graph) == 1);
    }
    assert(live.empty());
    std::cout << "allocator graph protection and OOM retry: PASS\n";
}
