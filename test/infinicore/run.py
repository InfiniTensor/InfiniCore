import os
import sys
import argparse
from pathlib import Path
import importlib.util

from framework import get_hardware_args_group


def find_ops_directory(location=None):
    """
    Find the ops directory by searching from location upwards.

    Args:
        location: Starting directory for search (default: current file's parent)

    Returns:
        Path: Path to ops directory or None if not found
    """
    if location is None:
        location = Path(__file__).parent / "ops"

    ops_dir = location.resolve()
    if ops_dir.exists() and any(ops_dir.glob("*.py")):
        return ops_dir

    return None


def get_available_operators(ops_dir):
    """
    Get list of available operators from ops directory.

    Args:
        ops_dir: Path to ops directory

    Returns:
        List of operator names
    """
    if not ops_dir or not ops_dir.exists():
        return []

    test_files = list(ops_dir.glob("*.py"))
    current_script = Path(__file__).name
    test_files = [f for f in test_files if f.name != current_script]

    operators = []
    for test_file in test_files:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                ):
                    operators.append(test_file.stem)
        except:
            continue

    return sorted(operators)


def import_operator_test(test_file_path):
    """
    Import an operator test module and return the test class instance.

    Args:
        test_file_path: Path to the test file

    Returns:
        tuple: (success, test_instance_or_error)
    """
    try:
        # Create a unique module name
        module_name = f"op_test_{test_file_path.stem}"

        # Load the module from file
        spec = importlib.util.spec_from_file_location(module_name, test_file_path)
        if spec is None or spec.loader is None:
            return False, f"Could not load module from {test_file_path}"

        module = importlib.util.module_from_spec(spec)

        # Add the module to sys.modules
        sys.modules[module_name] = module

        # Execute the module
        spec.loader.exec_module(module)

        # Find the test class (usually named OpTest)
        test_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and hasattr(attr, "__bases__")
                and any("BaseOperatorTest" in str(base) for base in attr.__bases__)
            ):
                test_class = attr
                break

        if test_class is None:
            return False, f"No test class found in {test_file_path}"

        # Create an instance
        test_instance = test_class()
        return True, test_instance

    except Exception as e:
        return False, f"Error importing {test_file_path}: {str(e)}"


def run_all_op_tests(
    ops_dir=None, specific_ops=None, bench=False, bench_mode="both", verbose=False
):
    """
    Run all operator test scripts in the ops directory using direct import.

    Args:
        ops_dir (str, optional): Path to the ops directory. If None, uses auto-detection.
        specific_ops (list, optional): List of specific operator names to test.
        bench (bool): Whether benchmarking is enabled
        bench_mode (str): Benchmark mode - "host", "device", or "both"
        verbose (bool): Whether verbose mode is enabled

    Returns:
        dict: Results dictionary with test names as keys and (success, test_runner, stdout, stderr) as values.
    """
    if ops_dir is None:
        ops_dir = find_ops_directory()
    else:
        ops_dir = Path(ops_dir)

    if not ops_dir or not ops_dir.exists():
        print(f"Error: Ops directory '{ops_dir}' does not exist.")
        return {}

    print(f"Looking for test files in: {ops_dir}")

    # Find all Python test files
    test_files = list(ops_dir.glob("*.py"))

    # Filter out this script itself and non-operator test files
    current_script = Path(__file__).name
    test_files = [f for f in test_files if f.name != current_script]

    # Filter to include only files that look like operator tests
    operator_test_files = []
    for test_file in test_files:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Look for characteristic patterns of operator tests
                if "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                ):
                    operator_test_files.append(test_file)
        except Exception as e:
            continue

    # Filter for specific operators if requested
    if specific_ops:
        filtered_files = []
        for test_file in operator_test_files:
            test_name = test_file.stem.lower()
            if any(op.lower() == test_name for op in specific_ops):
                filtered_files.append(test_file)
        operator_test_files = filtered_files

    if not operator_test_files:
        print(f"No operator test files found in {ops_dir}")
        print(f"Available Python files: {[f.name for f in test_files]}")
        return {}

    print(f"Found {len(operator_test_files)} operator test files:")
    for test_file in operator_test_files:
        print(f"  - {test_file.name}")

    results = {}

    cumulative_timing = {
        "total_torch_host_time": 0.0,
        "total_torch_device_time": 0.0,
        "total_infinicore_host_time": 0.0,
        "total_infinicore_device_time": 0.0,
        "operators_tested": 0,
    }

    for test_file in operator_test_files:
        test_name = test_file.stem

        try:
            # Import and run the test directly
            success, test_instance_or_error = import_operator_test(test_file)

            if not success:
                print(f"💥 {test_name}: ERROR - {test_instance_or_error}")
                results[test_name] = {
                    "success": False,
                    "return_code": -1,
                    "torch_host_time": 0.0,
                    "torch_device_time": 0.0,
                    "infini_host_time": 0.0,
                    "infini_device_time": 0.0,
                    "error_message": test_instance_or_error,
                    "test_runner": None,
                    "stdout": "",
                    "stderr": test_instance_or_error,
                }
                continue

            # Get the test runner class from the module
            test_module = sys.modules[f"op_test_{test_file.stem}"]
            if not hasattr(test_module, "GenericTestRunner"):
                print(f"💥 {test_name}: ERROR - No GenericTestRunner found")
                results[test_name] = {
                    "success": False,
                    "return_code": -1,
                    "torch_host_time": 0.0,
                    "torch_device_time": 0.0,
                    "infini_host_time": 0.0,
                    "infini_device_time": 0.0,
                    "error_message": "No GenericTestRunner found",
                    "test_runner": None,
                    "stdout": "",
                    "stderr": "No GenericTestRunner found",
                }
                continue

            # Create and run the test runner
            test_runner_class = test_module.GenericTestRunner
            runner_instance = test_runner_class(test_instance_or_error.__class__)

            # Temporarily redirect stdout to capture output
            from io import StringIO

            stdout_capture = StringIO()
            stderr_capture = StringIO()

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                # Run the test
                test_success, test_runner = runner_instance.run()

                # Get captured output
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()

                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                # Print the captured output
                if stdout_output:
                    print(stdout_output.rstrip())
                if stderr_output:
                    print("\nSTDERR:")
                    print(stderr_output.rstrip())

                # Analyze test results
                test_results = test_runner.get_test_results() if test_runner else []

                # Determine overall test status
                if test_success:
                    return_code = 0
                    status_icon = "✅"
                    status_text = "PASSED"
                else:
                    # Check if there are any failed tests
                    has_failures = any(
                        result.return_code == -1 for result in test_results
                    )
                    has_partial = any(
                        result.return_code == -3 for result in test_results
                    )
                    has_skipped = any(
                        result.return_code == -2 for result in test_results
                    )

                    if has_failures:
                        return_code = -1
                        status_icon = "❌"
                        status_text = "FAILED"
                    elif has_partial:
                        return_code = -3
                        status_icon = "⚠️"
                        status_text = "PARTIAL"
                    elif has_skipped:
                        return_code = -2
                        status_icon = "⏭️"
                        status_text = "SKIPPED"
                    else:
                        return_code = -1
                        status_icon = "❌"
                        status_text = "FAILED"

                # Calculate timing for all four metrics
                torch_host_time = sum(result.torch_host_time for result in test_results)
                torch_device_time = sum(
                    result.torch_device_time for result in test_results
                )
                infini_host_time = sum(
                    result.infini_host_time for result in test_results
                )
                infini_device_time = sum(
                    result.infini_device_time for result in test_results
                )

                results[test_name] = {
                    "success": test_success,
                    "return_code": return_code,
                    "torch_host_time": torch_host_time,
                    "torch_device_time": torch_device_time,
                    "infini_host_time": infini_host_time,
                    "infini_device_time": infini_device_time,
                    "error_message": "",
                    "test_runner": test_runner,
                    "stdout": stdout_output,
                    "stderr": stderr_output,
                }

                print(
                    f"{status_icon}  {test_name}: {status_text} (return code: {return_code})"
                )

                # Extract benchmark timing if in bench mode
                if bench and test_success and return_code == 0:
                    cumulative_timing["total_torch_host_time"] += torch_host_time
                    cumulative_timing["total_torch_device_time"] += torch_device_time
                    cumulative_timing["total_infinicore_host_time"] += infini_host_time
                    cumulative_timing[
                        "total_infinicore_device_time"
                    ] += infini_device_time
                    cumulative_timing["operators_tested"] += 1

            except Exception as e:
                # Restore stdout/stderr in case of exception
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                raise e

            # In verbose mode, stop execution on first failure
            if verbose and not test_success and return_code != 0:
                break

        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
            results[test_name] = {
                "success": False,
                "return_code": -1,
                "torch_host_time": 0.0,
                "torch_device_time": 0.0,
                "infini_host_time": 0.0,
                "infini_device_time": 0.0,
                "error_message": str(e),
                "test_runner": None,
                "stdout": "",
                "stderr": str(e),
            }

            # In verbose mode, stop execution on any exception
            if verbose:
                print(f"\n{'!'*60}")
                print(
                    f"VERBOSE MODE: Stopping execution due to exception in {test_name}"
                )
                print(f"{'!'*60}")
                break

    return results, cumulative_timing


def print_summary(
    results,
    verbose=False,
    total_expected_tests=0,
    cumulative_timing=None,
    bench_mode="both",
):
    """Print a comprehensive summary of test results including benchmark data."""
    print(f"\n{'='*80}")
    print("CUMULATIVE TEST SUMMARY")
    print(f"{'='*80}")

    if not results:
        print("No tests were run.")
        return False

    # Count different types of results
    passed = 0
    failed = 0
    skipped = 0
    partial = 0
    passed_operators = []  # Store passed operator names
    failed_operators = []  # Store failed operator names
    skipped_operators = []  # Store skipped operator names
    partial_operators = []  # Store partial operator names

    for test_name, result_data in results.items():
        return_code = result_data["return_code"]
        if return_code == 0:
            passed += 1
            passed_operators.append(test_name)
        elif return_code == -2:  # Special code for skipped tests
            skipped += 1
            skipped_operators.append(test_name)
        elif return_code == -3:  # Special code for partial tests
            partial += 1
            partial_operators.append(test_name)
        else:
            failed += 1
            failed_operators.append(test_name)

    total = len(results)

    print(f"Total tests run: {total}")
    if total_expected_tests > 0 and total < total_expected_tests:
        print(f"Total tests expected: {total_expected_tests}")
        print(f"Tests not executed: {total_expected_tests - total}")

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if skipped > 0:
        print(f"Skipped: {skipped}")

    if partial > 0:
        print(f"Partial: {partial}")

    # Print benchmark summary if cumulative_timing data is available
    if cumulative_timing and cumulative_timing["operators_tested"] > 0:
        print(f"{'-'*40}")
        print("BENCHMARK SUMMARY:")
        print(f"  Operators Tested: {cumulative_timing['operators_tested']}")

        # Display timing based on bench_mode
        if bench_mode in ["host", "both"]:
            print(
                f"  PyTorch    Host Total Time:   {cumulative_timing['total_torch_host_time']:12.3f} ms"
            )
            print(
                f"  InfiniCore Host Total Time:   {cumulative_timing['total_infinicore_host_time']:12.3f} ms"
            )

        if bench_mode in ["device", "both"]:
            print(
                f"  PyTorch    Device Total Time: {cumulative_timing['total_torch_device_time']:12.3f} ms"
            )
            print(
                f"  InfiniCore Device Total Time: {cumulative_timing['total_infinicore_device_time']:12.3f} ms"
            )

        print(f"{'-'*40}")

    # Display passed operators
    if passed_operators:
        print(f"\n✅ PASSED OPERATORS ({len(passed_operators)}):")
        # Display operators in groups of 10 per line
        for i in range(0, len(passed_operators), 10):
            line_ops = passed_operators[i : i + 10]
            print("  " + ", ".join(line_ops))
    else:
        print(f"\n✅ PASSED OPERATORS: None")

    # Display failed operators (if any)
    if failed_operators:
        print(f"\n❌ FAILED OPERATORS ({len(failed_operators)}):")
        for i in range(0, len(failed_operators), 10):
            line_ops = failed_operators[i : i + 10]
            print("  " + ", ".join(line_ops))

    # Display skipped operators (if any)
    if skipped_operators:
        print(f"\n⏭️ SKIPPED OPERATORS ({len(skipped_operators)}):")
        for i in range(0, len(skipped_operators), 10):
            line_ops = skipped_operators[i : i + 10]
            print("  " + ", ".join(line_ops))

    # Display partial operators (if any)
    if partial_operators:
        print(f"\n⚠️  PARTIAL OPERATORS ({len(partial_operators)}):")
        for i in range(0, len(partial_operators), 10):
            line_ops = partial_operators[i : i + 10]
            print("  " + ", ".join(line_ops))

    if total > 0:
        # Calculate success rate based on actual executed tests
        executed_tests = passed + failed + partial
        if executed_tests > 0:
            success_rate = passed / executed_tests * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")

    if verbose and total < total_expected_tests:
        print(f"\n💡 Verbose mode: Execution stopped after first failure")
        print(f"   {total_expected_tests - total} tests were not executed")

    if failed == 0:
        if skipped > 0 or partial > 0:
            print(f"\n⚠️  Tests completed with some operators not implemented")
            print(f"   - {skipped} tests skipped (both operators not implemented)")
            print(f"   - {partial} tests partial (one operator not implemented)")
        else:
            print(f"\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False


def list_available_tests(ops_dir=None):
    """List all available operator test files."""
    if ops_dir is None:
        ops_dir = find_ops_directory()
    else:
        ops_dir = Path(ops_dir)

    if not ops_dir or not ops_dir.exists():
        print(f"Error: Ops directory '{ops_dir}' does not exist.")
        return

    operators = get_available_operators(ops_dir)

    if operators:
        print(f"Available operator test files in {ops_dir}:")
        for operator in operators:
            print(f"  - {operator}")
        print(f"\nTotal: {len(operators)} operators")
    else:
        print(f"No operator test files found in {ops_dir}")
        # Show available Python files for debugging
        test_files = list(ops_dir.glob("*.py"))
        current_script = Path(__file__).name
        test_files = [f for f in test_files if f.name != current_script]
        if test_files:
            print(f"Available Python files: {[f.name for f in test_files]}")


def generate_help_epilog(ops_dir):
    """
    Generate dynamic help epilog with available operators and hardware platforms.

    Args:
        ops_dir: Path to ops directory

    Returns:
        str: Formatted help text
    """
    # Get available operators
    operators = get_available_operators(ops_dir)

    # Build epilog text
    epilog_parts = []

    # Examples section
    epilog_parts.append("Examples:")
    epilog_parts.append("  # Run all operator tests on CPU")
    epilog_parts.append("  python run.py --cpu")
    epilog_parts.append("")
    epilog_parts.append("  # Run specific operators")
    epilog_parts.append("  python run.py --ops add matmul --nvidia")
    epilog_parts.append("")
    epilog_parts.append("  # Run with debug mode on multiple devices")
    epilog_parts.append("  python run.py --cpu --nvidia --debug")
    epilog_parts.append("")
    epilog_parts.append(
        "  # Run with verbose mode to stop on first error with full traceback"
    )
    epilog_parts.append("  python run.py --cpu --nvidia --verbose")
    epilog_parts.append("")
    epilog_parts.append("  # Run with benchmarking (both host and device timing)")
    epilog_parts.append("  python run.py --cpu --bench")
    epilog_parts.append("")
    epilog_parts.append("  # Run with host timing only")
    epilog_parts.append("  python run.py --nvidia --bench host")
    epilog_parts.append("")
    epilog_parts.append("  # Run with device timing only")
    epilog_parts.append("  python run.py --nvidia --bench device")
    epilog_parts.append("")
    epilog_parts.append("  # List available tests without running")
    epilog_parts.append("  python run.py --list")
    epilog_parts.append("")

    # Available operators section
    if operators:
        epilog_parts.append("Available Operators:")
        # Group operators for better display
        operators_per_line = 4
        for i in range(0, len(operators), operators_per_line):
            line_ops = operators[i : i + operators_per_line]
            epilog_parts.append(f"  {', '.join(line_ops)}")
        epilog_parts.append("")
    else:
        epilog_parts.append("Available Operators: (none detected)")
        epilog_parts.append("")

    # Additional notes
    epilog_parts.append("Note:")
    epilog_parts.append(
        "  - Use '--' to pass additional arguments to individual test scripts"
    )
    epilog_parts.append(
        "  - Operators are automatically discovered from the ops directory"
    )
    epilog_parts.append(
        "  - --bench mode now shows cumulative timing across all operators"
    )
    epilog_parts.append(
        "  - --bench host/device/both controls host/device timing measurement"
    )
    epilog_parts.append(
        "  - --verbose mode stops execution on first error and shows full traceback"
    )

    return "\n".join(epilog_parts)


def main():
    """Main entry point with comprehensive command line argument parsing."""
    # First, find ops directory for dynamic help generation
    ops_dir = find_ops_directory()

    parser = argparse.ArgumentParser(
        description="Run InfiniCore operator tests across multiple hardware platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_epilog(ops_dir),
    )

    # Core options
    parser.add_argument(
        "--ops-dir", type=str, help="Path to the ops directory (default: auto-detect)"
    )
    parser.add_argument(
        "--ops", nargs="+", help="Run specific operators only (e.g., --ops add matmul)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test files without running them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode to stop on first error with full traceback",
    )
    parser.add_argument(
        "--bench",
        nargs="?",
        const="both",
        choices=["host", "device", "both"],
        help="Enable performance benchmarking mode. "
        "Options: host (CPU time only), device (GPU time only), both (default)",
    )

    get_hardware_args_group(parser)

    # Parse known args first, leave the rest for the test scripts
    args, unknown_args = parser.parse_known_args()

    # Handle list command
    if args.list:
        list_available_tests(args.ops_dir)
        return

    # Auto-detect ops directory if not provided
    if args.ops_dir is None:
        ops_dir = find_ops_directory()
        if not ops_dir:
            print(
                "Error: Could not auto-detect ops directory. Please specify with --ops-dir"
            )
            sys.exit(1)
    else:
        ops_dir = Path(args.ops_dir)
        if not ops_dir.exists():
            print(f"Error: Ops directory '{ops_dir}' does not exist.")
            sys.exit(1)

    # Show what extra arguments will be passed
    if unknown_args:
        print(f"Passing extra arguments to test scripts: {unknown_args}")

    # Get available operators for display
    available_operators = get_available_operators(ops_dir)

    print(f"InfiniCore Operator Test Runner")
    print(f"Operating directory: {ops_dir}")
    print(f"Available operators: {len(available_operators)}")

    if args.verbose:
        print(f"Verbose mode: ENABLED (will stop on first error with full traceback)")

    if args.bench:
        bench_mode = args.bench if args.bench != "both" else "both"
        print(f"Benchmark mode: {bench_mode.upper()} timing")

    if args.ops:
        # Validate requested operators
        valid_ops = []
        invalid_ops = []
        for op in args.ops:
            if op in available_operators:
                valid_ops.append(op)
            else:
                invalid_ops.append(op)

        if invalid_ops:
            print(f"Warning: Unknown operators: {', '.join(invalid_ops)}")
            print(f"Available operators: {', '.join(available_operators)}")

        if valid_ops:
            print(f"Testing operators: {', '.join(valid_ops)}")
            total_expected_tests = len(valid_ops)
        else:
            print("No valid operators specified. Running all available tests.")
            total_expected_tests = len(available_operators)
    else:
        print("Testing all available operators")
        total_expected_tests = len(available_operators)

    print()

    # Run all tests
    results, cumulative_timing = run_all_op_tests(
        ops_dir=ops_dir,
        specific_ops=args.ops,
        bench=bool(args.bench),
        bench_mode=args.bench if args.bench else "both",
        verbose=args.verbose,
    )

    # Print summary and exit with appropriate code
    all_passed = print_summary(
        results,
        args.verbose,
        total_expected_tests,
        cumulative_timing,
        bench_mode=args.bench if args.bench else "both",
    )

    # Check if there were any tests with missing implementations
    has_missing_implementations = any(
        result_data["return_code"] in [-2, -3] for result_data in results.values()
    )

    if all_passed and has_missing_implementations:
        print(f"\n⚠️  Note: Some operators are not fully implemented")
        print(f"   Run individual tests for details on missing implementations")

    if args.verbose and not all_passed:
        print(
            f"\n💡 Verbose mode tip: Use individual test commands for detailed debugging:"
        )
        failed_ops = [
            name
            for name, result_data in results.items()
            if result_data["return_code"] == -1
        ]
        for op in failed_ops[:3]:  # Show first 3 failed operators
            print(f"   python {ops_dir / (op + '.py')} --verbose")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
