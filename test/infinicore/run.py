import sys
import argparse
from pathlib import Path

# Import components from the unified framework package
from framework.driver import TestDriver
from framework.summary import TestSummary
from framework.structs import TestTiming
from framework import get_hardware_args_group, add_common_test_args


class TestDiscoverer:
    def __init__(self, ops_dir_path=None):
        self.ops_dir = self._resolve_dir(ops_dir_path)

    def _resolve_dir(self, path):
        if path:
            p = Path(path)
            if p.exists():
                return p

        # Default fallback logic: 'ops' directory under the parent of the current file's parent.
        # Note: Since this file is in 'infinicore/', we look at parent.
        # It is recommended to pass an explicit path in run.py.
        fallback = Path(__file__).parent / "ops"
        return fallback if fallback.exists() else None

    def get_available_operators(self):
        """Returns a list of names of all available operators."""
        if not self.ops_dir:
            return []
        files = self.scan()
        return sorted([f.stem for f in files])

    def get_raw_python_files(self):
        """
        Get all .py files in the directory (excluding run.py) without content validation.
        Used for debugging: helps identify files that exist but failed validation.
        """
        if not self.ops_dir or not self.ops_dir.exists():
            return []

        files = list(self.ops_dir.glob("*.py"))
        # Exclude run.py itself and __init__.py
        return [
            f.name for f in files if f.name != "run.py" and not f.name.startswith("__")
        ]

    def scan(self, specific_ops=None):
        """Scans and returns a list of Path objects that meet the criteria."""
        if not self.ops_dir or not self.ops_dir.exists():
            return []

        # 1. Find all .py files
        files = list(self.ops_dir.glob("*.py"))

        target_ops_set = set(specific_ops) if specific_ops else None

        # 2. Filter out non-test files (via content check)
        valid_files = []
        for f in files:
            # A. Basic Name Filtering
            if f.name.startswith("_") or f.name == "run.py":
                continue

            # B. Specific Ops Filtering
            if target_ops_set and f.stem not in target_ops_set:
                continue

            # C. Content Check (Expensive I/O)
            # Only perform this check if the file passed the name filters above.
            if self._is_operator_test(f):
                valid_files.append(f)

        return valid_files

    def _is_operator_test(self, file_path):
        """Checks if the file content contains operator test characteristics."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                )
        except:
            return False


def generate_help_epilog(ops_dir=None):
    """
    Generate dynamic help epilog containing available operators and hardware platforms.
    Maintains the original output format for backward compatibility.
    """
    # === Adapter: Use TestDiscoverer to get operator list ===
    # Temporarily instantiate a Discoverer just to fetch the list
    discoverer = TestDiscoverer(ops_dir)
    operators = discoverer.get_available_operators()

    # Build epilog text (fully replicating original logic)
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
    """Main entry point for the InfiniCore Operator Test Runner."""
    parser = argparse.ArgumentParser(
        description="Run InfiniCore operator tests across multiple hardware platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_epilog(),
    )
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

    # Add common test arguments (including --save, --bench, etc.)
    add_common_test_args(parser)
    get_hardware_args_group(parser)

    args, unknown_args = parser.parse_known_args()
    # Show what extra arguments will be passed
    if unknown_args:
        print(f"Passing extra arguments to test scripts: {unknown_args}")

    # 1. Discovery
    discoverer = TestDiscoverer(args.ops_dir)
    if args.list:
        print("Available operators:", discoverer.get_available_operators())
        return

    if args.verbose:
        print(f"Verbose mode: ENABLED (will stop on first error with full traceback)")

    if args.bench:
        print(f"Benchmark mode: {args.bench.upper()} timing")

    target_ops = None

    if args.ops:
        # Get all available operator names
        available_ops = set(discoverer.get_available_operators())
        requested_ops = set(args.ops)

        # Classify using set operations
        valid_ops = list(requested_ops & available_ops)  # Intersection: Valid ops
        invalid_ops = list(requested_ops - available_ops)  # Difference: Invalid ops

        # Warn if there are invalid operators
        if invalid_ops:
            print(f"⚠️  Warning: The following requested operators were not found:")
            print(f"   {', '.join(invalid_ops)}")
            print(f"   (Use --list to see available operators)")

        if not valid_ops:
            # Case A: User input provided, but ALL were invalid.
            print(f"⚠️  No valid operators remained from your list.")
            print(f"🔄 Fallback: Proceeding to run ALL available tests...")

        else:
            # Case B: At least some valid operators found.
            print(f"🎯 Targeted operators: {', '.join(valid_ops)}")
            target_ops = valid_ops

    test_files = discoverer.scan(target_ops)
    if not test_files:
        print("No tests found.")
        sys.exit(0)

    # 2. Preparation
    driver = TestDriver()
    cumulative_timing = TestTiming()
    test_summary = TestSummary(args.verbose, args.bench)
    # printer = ConsolePrinter()
    results = []

    test_summary.print_header(discoverer.ops_dir, len(test_files))
    # printer.print_header(discoverer.ops_dir, len(test_files))

    # 3. Execution Loop
    for f in test_files:
        result = driver.drive(f)
        results.append(result)

        # Real-time reporting and printing of stdout
        test_summary.print_live_result(result)

        # Accumulate timing
        if result.success:
            cumulative_timing.torch_host += result.timing.torch_host
            cumulative_timing.infini_host += result.timing.infini_host
            cumulative_timing.torch_device += result.timing.torch_device
            cumulative_timing.infini_device += result.timing.infini_device
            cumulative_timing.operators_tested += 1

        # Fail fast in verbose mode
        if args.verbose and not result.success:
            print("\nStopping due to failure in verbose mode.")
            break

    # 4. Final Report & Save
    all_passed = test_summary.print_summary(
        results,
        cumulative_timing if args.bench else None,
        ops_dir=discoverer.ops_dir,
        total_expected=len(test_files),
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
