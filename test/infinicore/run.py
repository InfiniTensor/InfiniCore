import os
import sys
import subprocess
import argparse
from pathlib import Path


def find_ops_directory(start_dir=None):
    """
    Find the ops directory by searching from start_dir upwards.
    """
    if start_dir is None:
        start_dir = Path(__file__).parent

    ops_dir = start_dir / "ops"
    if ops_dir.exists() and (ops_dir / "rms_norm.py").exists():
        return ops_dir


def run_all_op_tests(ops_dir=None, verbose=False, specific_ops=None, extra_args=None):
    """
    Run all operator test scripts in the ops directory.

    Args:
        ops_dir (str, optional): Path to the ops directory. If None, uses the current directory.
        verbose (bool): Whether to print detailed output.
        specific_ops (list, optional): List of specific operator names to test (e.g., ['add', 'matmul']).
        extra_args (list, optional): Extra command line arguments to pass to test scripts.

    Returns:
        dict: Results dictionary with test names as keys and (success, return_code, output) as values.
    """
    if ops_dir is None:
        ops_dir = find_ops_directory()
    else:
        ops_dir = Path(ops_dir)

    if not ops_dir.exists():
        print(f"Error: Ops directory '{ops_dir}' does not exist.")
        return {}

    print(f"Looking for test files in: {ops_dir}")

    # Find all Python test files (looking for actual operator test files)
    test_files = list(ops_dir.glob("*.py"))

    # Filter out this script itself and non-operator test files
    current_script = Path(__file__).name
    test_files = [f for f in test_files if f.name != current_script]

    # Further filter to include only files that look like operator tests
    # (they typically import infinicore and BaseOperatorTest)
    operator_test_files = []
    for test_file in test_files:
        try:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "infinicore" in content and "BaseOperatorTest" in content:
                    operator_test_files.append(test_file)
                elif verbose:
                    print(f"  Skipping {test_file.name}: not an operator test file")
        except Exception as e:
            if verbose:
                print(f"  Could not read {test_file.name}: {e}")
            continue

    if specific_ops:
        # Filter for specific operators (case insensitive)
        filtered_files = []
        for test_file in operator_test_files:
            test_name = test_file.stem.lower()
            if any(op.lower() in test_name for op in specific_ops):
                filtered_files.append(test_file)
            elif verbose:
                print(f"  Filtered out {test_file.name}: not in specific_ops list")
        operator_test_files = filtered_files

    if not operator_test_files:
        print(f"No operator test files found in {ops_dir}")
        print(f"Available Python files: {[f.name for f in test_files]}")
        print(f"Current directory: {Path.cwd()}")
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

            if verbose:
                print(f"Command: {' '.join(cmd)}")
                print(f"Working directory: {ops_dir}")

            # Always capture output to display it
            result = subprocess.run(cmd, cwd=ops_dir, capture_output=True, text=True)

            success = result.returncode == 0
            results[test_name] = (
                success,
                result.returncode,
                result.stdout,
                result.stderr,
            )

            # Print the output from the test script
            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            if success:
                print(f"✅ {test_name}: PASSED (return code: {result.returncode})")
            else:
                print(f"❌ {test_name}: FAILED (return code: {result.returncode})")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = (False, -1, "", str(e))

    return results


def print_summary(results):
    """Print a summary of test results."""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    if not results:
        print("No tests were run.")
        return

    passed = sum(1 for success, _, _, _ in results.values() if success)
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if total > 0:
        print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\nFailed tests:")
        for test_name, (success, returncode, stdout, stderr) in results.items():
            if not success:
                print(f"  - {test_name} (return code: {returncode})")
                # Print brief error info for failed tests
                if stderr:
                    error_lines = stderr.strip().split("\n")
                    if error_lines:
                        print(f"    Error: {error_lines[0]}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run all operator tests in the ops directory", add_help=False
    )

    # Our script's specific arguments
    parser.add_argument(
        "--ops-dir", type=str, help="Path to the ops directory (default: auto-detect)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed command information for each test",
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
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )

    # Parse known args first, leave the rest for the test scripts
    args, unknown_args = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print("\nExtra arguments that will be passed to test scripts:")
        print("  --nvidia, --cpu, --bench, --debug, etc.")
        return

    # Auto-detect ops directory if not provided
    if args.ops_dir is None:
        ops_dir = find_ops_directory()
    else:
        ops_dir = Path(args.ops_dir)

    if args.list:
        # Just list available test files
        test_files = list(ops_dir.glob("*.py"))
        current_script = Path(__file__).name
        test_files = [f for f in test_files if f.name != current_script]

        operator_test_files = []
        for test_file in test_files:
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "infinicore" in content and "BaseOperatorTest" in content:
                        operator_test_files.append(test_file)
            except:
                continue

        if operator_test_files:
            print(f"Available operator test files in {ops_dir}:")
            for test_file in operator_test_files:
                print(f"  - {test_file.name}")
        else:
            print(f"No operator test files found in {ops_dir}")
            print(f"Available Python files: {[f.name for f in test_files]}")
        return

    # Show what extra arguments will be passed
    if unknown_args:
        print(f"Passing extra arguments to test scripts: {unknown_args}")

    # Run all tests
    results = run_all_op_tests(
        ops_dir=ops_dir,
        verbose=args.verbose,
        specific_ops=args.ops,
        extra_args=unknown_args,
    )

    print_summary(results)

    # Exit with appropriate code
    if results and all(success for success, _, _, _ in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
