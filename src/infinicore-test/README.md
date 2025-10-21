# InfiniCore Memory Management Test Suite

This test suite provides comprehensive testing for the InfiniCore memory management system, focusing on the critical issues identified in the memory management architecture analysis.

## Overview

The test suite includes six main test categories:

1. **Basic Memory Tests** - Basic allocation, deallocation, and memory operations
2. **Concurrency Tests** - Thread safety and concurrent access testing
3. **Exception Safety Tests** - Exception handling and safety testing
4. **Memory Leak Tests** - Memory leak detection and prevention
5. **Performance Tests** - Performance benchmarks and optimization validation
6. **Stress Tests** - High-load stress testing and edge cases

## Building

### Using XMake (if integrated with main build)
```bash
# From InfiniCore root directory
xmake build infinicore-test
```

## Running Tests

### Run All Tests
```bash
./infinicore-test
```

### Run Specific Test Categories
```bash
# Basic memory tests
./infinicore-test --test basic

# Concurrency tests
./infinicore-test --test concurrency

# Exception safety tests
./infinicore-test --test exception

# Memory leak tests
./infinicore-test --test leak

# Performance tests
./infinicore-test --test performance

# Stress tests
./infinicore-test --test stress
```

### Run with Specific Device
```bash
# Run on CPU
./infinicore-test --cpu

# Run on NVIDIA GPU
./infinicore-test --nvidia

# Run on other devices
./infinicore-test --cambricon
./infinicore-test --ascend
./infinicore-test --metax
./infinicore-test --moore
./infinicore-test --iluvatar
./infinicore-test --kunlun
./infinicore-test --hygon
```

### Customize Test Parameters
```bash
# Run with custom thread count
./infinicore-test --threads 8

# Run with custom iteration count
./infinicore-test --iterations 5000

# Combine options
./infinicore-test --nvidia --test concurrency --threads 16 --iterations 2000
```

## Test Categories

### 1. Basic Memory Tests
Tests fundamental memory operations:
- Memory allocation and deallocation
- Memory size and device properties
- Memory read/write operations
- Pinned memory allocation
- Memory data integrity

### 2. Concurrency Tests
Tests thread safety and concurrent access:
- **Concurrent Allocations**: Multiple threads allocating memory simultaneously
- **Concurrent Device Switching**: Multiple threads switching device contexts
- **Memory Allocation Race**: Race condition testing for memory operations

### 3. Exception Safety Tests
Tests exception handling and safety:
- **Allocation Failure**: Tests behavior when allocation fails
- **Deallocation Exception**: Tests exception safety during deallocation
- **Context Switch Exception**: Tests exception handling during device switching

### 4. Memory Leak Tests
Tests memory leak detection and prevention:
- **Basic Leak Detection**: Basic memory leak detection
- **Cross-Device Leak Detection**: Memory leaks in cross-device scenarios
- **Exception Leak Detection**: Memory leaks during exception handling

### 5. Performance Tests
Tests performance and benchmarks:
- **Allocation Performance**: Memory allocation speed benchmarks
- **Concurrent Performance**: Performance under concurrent load
- **Memory Copy Performance**: Memory copy bandwidth tests

### 6. Stress Tests
Tests high-load scenarios and edge cases:
- **High Frequency Allocations**: Rapid allocation/deallocation cycles
- **Large Memory Allocations**: Large memory block allocation
- **Cross-Device Stress**: Stress testing across multiple devices

## Expected Results

### Critical Issues to Watch For

The tests are designed to detect the critical issues identified in the memory management analysis:

1. **Thread Safety Violations**
   - Race conditions in concurrent allocations
   - Inconsistent device context switching
   - Global state corruption

2. **Memory Leaks**
   - Unfreed memory after deallocation
   - Cross-device memory not properly cleaned up
   - Exception-related memory leaks

3. **Exception Safety Issues**
   - Exceptions during allocation causing resource leaks
   - Exceptions in destructors causing `std::terminate`
   - Incomplete cleanup on exceptions

4. **Performance Issues**
   - Slow allocation/deallocation performance
   - Poor concurrent performance
   - Inefficient memory copy operations

### Performance Thresholds

The tests include performance thresholds:

- **Allocation Performance**: < 100μs per allocation
- **Concurrent Performance**: < 200μs per allocation under load
- **Memory Bandwidth**: > 100 MB/s for memory copies

## Test Output

### Successful Test Run
```
==============================================
InfiniCore Memory Management Test Suite
==============================================
Device: 0
Threads: 4
Iterations: 1000
==============================================

[SUITE] Running: BasicMemoryTest
[TEST] Starting: BasicMemoryTest
[TEST] PASSED: BasicMemoryTest (Duration: 1234μs)

[SUITE] Running: ConcurrencyTest
[TEST] Starting: ConcurrencyTest
[TEST] PASSED: ConcurrencyTest (Duration: 5678μs)

...

==============================================
Test Summary
==============================================
Total Tests: 6
Passed: 6
Failed: 0
Total Time: 12345μs
==============================================

✅ All tests passed!
```

### Failed Test Run
```
[TEST] FAILED: ConcurrencyTest - Concurrent allocation test failed: expected 8000 successes, got 7995 successes and 5 failures

==============================================
Final Results
==============================================
Total Tests: 6
Passed: 5
Failed: 1
==============================================

❌ Some tests failed. Please review the output above.
```

## Debugging Failed Tests

### Common Issues and Solutions

1. **Thread Safety Failures**
   - Check for race conditions in global state access
   - Verify proper synchronization in allocators
   - Review device context switching logic

2. **Memory Leak Failures**
   - Check deallocation logic in allocators
   - Verify cross-device cleanup mechanisms
   - Review exception safety in destructors

3. **Performance Failures**
   - Profile allocation/deallocation paths
   - Check for unnecessary context switching
   - Review memory copy implementations

4. **Exception Safety Failures**
   - Verify no-throw guarantees in destructors
   - Check exception handling in allocation paths
   - Review resource cleanup on exceptions

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run Memory Tests
  run: |
    cd src/infinicore-test
    mkdir build && cd build
    cmake ..
    make
    ./infinicore-test --test all
```

### Custom Test Targets
```bash
# Run specific test categories
make test-memory-basic
make test-memory-concurrency
make test-memory-exception
make test-memory-leak
make test-memory-performance
make test-memory-stress
make test-memory-all
```

## Contributing

When adding new tests:

1. Follow the existing test framework pattern
2. Add appropriate error messages and logging
3. Include performance thresholds where applicable
4. Test both success and failure scenarios
5. Update this README with new test descriptions

## Dependencies

- InfiniCore library (infinicore, infiniop, infinirt, infiniccl)
- C++17 compatible compiler
- Threading library (pthread on Linux)
- CMake 3.16+ (for CMake build)

## Notes

- Tests are designed to be deterministic where possible
- Some tests may have timing dependencies
- Performance tests may vary based on system load
- Memory leak detection is basic and may not catch all leaks
- Tests assume proper InfiniCore initialization
