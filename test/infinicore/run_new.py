import sys
import argparse
from pathlib import Path

# Import components from the unified framework package
from framework.loader import TestDiscoverer
from framework.executor import SingleTestExecutor
from framework.reporter import TestReporter
from framework.datatypes import TestTiming
from framework import get_hardware_args_group, add_common_test_args

def main():
    """Main entry point for the InfiniCore Operator Test Runner."""
    parser = argparse.ArgumentParser(description="Run InfiniCore operator tests across multiple hardware platforms")
    parser.add_argument("--ops-dir", type=str, help="Path to the ops directory (default: auto-detect)")
    parser.add_argument("--ops", nargs="+", help="Run specific operators only (e.g., --ops add matmul)")
    parser.add_argument("--list", action="store_true", help="List all available test files without running them")
    
    # Add common test arguments (including --save, --bench, etc.)
    add_common_test_args(parser)
    get_hardware_args_group(parser)
    
    args, _ = parser.parse_known_args()

    # 1. Discovery
    discoverer = TestDiscoverer(args.ops_dir)
    if args.list:
        print("Available operators:", discoverer.get_available_operators())
        return

    test_files = discoverer.scan(args.ops)
    if not test_files:
        print("No tests found.")
        sys.exit(0)

    # 2. Preparation
    executor = SingleTestExecutor()
    cumulative_timing = TestTiming()
    results = []
    
    TestReporter.print_header(discoverer.ops_dir, len(test_files))

    # 3. Execution Loop
    for f in test_files:
        result = executor.run(f)
        results.append(result)
        
        # Real-time reporting and printing of stdout
        TestReporter.print_live_result(result, verbose=args.verbose)

        # Accumulate timing
        if result.success:
            cumulative_timing.torch_host += result.timing.torch_host
            cumulative_timing.infini_host += result.timing.infini_host
            cumulative_timing.torch_device += result.timing.torch_device
            cumulative_timing.infini_device += result.timing.infini_device

        # Fail fast in verbose mode
        if args.verbose and not result.success:
            print("\nStopping due to failure in verbose mode.")
            break

    # 4. Final Report & Save
    all_passed = TestReporter.print_summary(
        results, 
        cumulative_timing if args.bench else None,
        total_expected=len(test_files)
    )

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
