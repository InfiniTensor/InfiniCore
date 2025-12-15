import sys
import argparse
from pathlib import Path

# 引入我们拆分的模块
from lib.discovery import TestDiscoverer
from lib.execution import SingleTestExecutor
from lib.reporting import ConsoleReporter
from lib.types import TestTiming

# 引入你原来的 framework (保持不动)
from framework import get_hardware_args_group, add_common_test_args

def main():
    parser = argparse.ArgumentParser(description="InfiniCore Test Runner")
    parser.add_argument("--ops-dir", type=str, help="Ops directory path")
    parser.add_argument("--ops", nargs="+", help="Specific operators to run")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    add_common_test_args(parser)
    get_hardware_args_group(parser)
    
    args, _ = parser.parse_known_args()

    # 1. Discovery (发现测试)
    discoverer = TestDiscoverer(args.ops_dir)
    if args.list:
        print("Available operators:", discoverer.get_available_operators())
        return

    test_files = discoverer.scan(args.ops)
    if not test_files:
        print("No tests found.")
        sys.exit(0)

    # 2. Preparation (准备工作)
    executor = SingleTestExecutor()
    reporter = ConsoleReporter()
    cumulative_timing = TestTiming()
    results = []
    
    reporter.print_header(discoverer.ops_dir, len(test_files))

    # 3. Execution Loop (执行循环)
    for f in test_files:
        # 执行
        result = executor.run(f)
        results.append(result)
        
        # 实时报告
        reporter.print_live_result(result, verbose=args.verbose)

        # 统计时间
        if result.success:
            cumulative_timing.torch_host += result.timing.torch_host
            cumulative_timing.infini_host += result.timing.infini_host
            # ... 其他时间累加 ...

        # 快速失败
        if args.verbose and not result.success:
            print("\nStopping due to failure in verbose mode.")
            break

    # 4. Final Report (最终报告)
    all_passed = reporter.print_summary(
        results, 
        cumulative_timing if args.bench else None,
        total_expected=len(test_files)
    )

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
