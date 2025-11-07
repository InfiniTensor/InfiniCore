import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Tuple, List


def find_ops_directory(start_dir=None):
    """
    Find the ops directory by searching from start_dir upwards.

    Args:
        start_dir: Starting directory for search (default: current file's parent)

    Returns:
        Path: Path to ops directory or None if not found
    """
    if start_dir is None:
        start_dir = Path(__file__).parent

    # Look for ops directory in common locations
    possible_locations = [
        start_dir / "ops",
        start_dir / ".." / "ops",
        start_dir / ".." / "test" / "ops",
        start_dir / "test" / "ops",
    ]

    for location in possible_locations:
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


def run_all_op_tests(ops_dir=None, specific_ops=None, extra_args=None):
    """
    Run all operator test scripts in the ops directory.

    Args:
        ops_dir (str, optional): Path to the ops directory. If None, uses auto-detection.
        specific_ops (list, optional): List of specific operator names to test.
        extra_args (list, optional): Extra command line arguments to pass to test scripts.

    Returns:
        dict: Results dictionary with test names as keys and (success, return_code, stdout, stderr) as values.
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
            if any(op.lower() in test_name for op in specific_ops):
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

    for test_file in operator_test_files:
        test_name = test_file.stem

        try:
            # Run the test script
            cmd = [sys.executable, str(test_file)]

            # Add extra arguments if provided
            if extra_args:
                cmd.extend(extra_args)

            result = subprocess.run(
                cmd,
                cwd=ops_dir,
                stdout=None,
                stderr=None,
            )

            success = result.returncode == 0
            results[test_name] = (
                success,
                result.returncode,
                "",
                "",
            )

            # Print the output from the test script
            print(f"\n{'='*60}")
            print(f"TEST: {test_name}")
            print(f"{'='*60}")

            if result.stdout:
                print(result.stdout.rstrip())

            if result.stderr:
                print("\nSTDERR:")
                print(result.stderr.rstrip())

            status_icon = "✅" if success else "❌"
            print(
                f"{status_icon} {test_name}: {'PASSED' if success else 'FAILED'} (return code: {result.returncode})"
            )

        except Exception as e:
            print(f"💥 {test_name}: ERROR - {str(e)}")
            results[test_name] = (False, -1, "", str(e))

    return results


def print_summary(results):
    """Print a comprehensive summary of test results."""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    if not results:
        print("No tests were run.")
        return False

    passed = sum(1 for success, _, _, _ in results.values() if success)
    total = len(results)
    failed_tests = [name for name, (success, _, _, _) in results.items() if not success]

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if total > 0:
        success_rate = passed / total * 100
        print(f"Success rate: {success_rate:.1f}%")

    if not failed_tests:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {len(failed_tests)} tests failed:")
        for test_name in failed_tests:
            success, returncode, stdout, stderr = results[test_name]
            print(f"  - {test_name} (return code: {returncode})")

            # Print brief error info for failed tests
            if stderr:
                error_lines = stderr.strip().split("\n")
                if error_lines:
                    # Take first meaningful error line
                    for line in error_lines:
                        if line.strip() and not line.startswith("Warning:"):
                            print(f"    Error: {line.strip()}")
                            break
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
        "  - --bench option is disabled in batch mode (run individual tests for benchmarking)"
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

    # Hardware platform options using shared function
    from framework import get_hardware_args_group

    hardware_group = get_hardware_args_group(parser)

    # Parse known args first, leave the rest for the test scripts
    args, unknown_args = parser.parse_known_args()

    # Handle list command
    if args.list:
        list_available_tests(args.ops_dir)
        return

    # Check for --bench option in extra arguments
    for arg in unknown_args:
        if arg in ["--bench"]:
            print("❌ ERROR: --bench option is not allowed in batch testing mode.")
            print("")
            print("Solution: Run individual test scripts for benchmarking:")
            print("          python path/to/individual_test.py --bench --<platform>")
            sys.exit(1)

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
        else:
            print("No valid operators specified. Running all available tests.")
    else:
        print("Testing all available operators")

    print()

    # Run all tests
    results = run_all_op_tests(
        ops_dir=ops_dir,
        specific_ops=args.ops,
        extra_args=unknown_args,
    )

    # Print summary and exit with appropriate code
    all_passed = print_summary(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
