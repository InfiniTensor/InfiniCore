import sys
import argparse
import json
import os
from pathlib import Path

from framework import (
    get_hardware_args_group,
    add_common_test_args,
    InfiniDeviceEnum,
    InfiniDeviceNames,
)
from framework.test_manager import TestCollector, TestManager


def generate_help_epilog(ops_dir=None):
    """
    Generate dynamic help epilog containing available operators and hardware platforms.
    Maintains the original output format for backward compatibility.
    """
    # === Adapter: Use TestCollector to get operator list ===
    # Temporarily instantiate a Collector just to fetch the list
    collector = TestCollector(ops_dir)
    operators = collector.get_available_operators()

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


def fill_defaults_for_local_mode(args):
    """
    Helper function specifically for Local Scan mode to fill default arguments.
    Since parser defaults are set to None (to handle override logic in load mode),
    we need to manually fill None with default values in local mode.
    """
    # 1. Copy args to avoid modifying the original object and affecting other logic
    # argparse.Namespace can be converted to dict and back, or copied directly
    local_args = argparse.Namespace(**vars(args))

    # 2. Fill default values for numeric arguments
    if local_args.num_prerun is None:
        local_args.num_prerun = 10

    if local_args.num_iterations is None:
        local_args.num_iterations = 1000

    return local_args


def load_and_override_cases(load_paths, args):
    """
    Load JSON, apply CLI overrides, and handle all argument logic.
    """
    cases = []
    files_to_read = []

    # 1. Scan
    for p_str in load_paths:
        p = Path(p_str)
        if p.is_dir():
            files_to_read.extend(p.glob("*.json"))
        elif p.is_file():
            files_to_read.append(p)

    # 2. Read and Validate
    loaded_count = 0
    skipped_count = 0

    for f_path in files_to_read:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Unify as a list to handle both single dict and list of dicts
                current_batch = data if isinstance(data, list) else [data]
                
                valid_batch = []
                for item in current_batch:
                    # We only require the 'operator' field to identify the test case.
                    if isinstance(item, dict) and "operator" in item:
                        valid_batch.append(item)
                    else:
                        skipped_count += 1
                
                if valid_batch:
                    cases.extend(valid_batch)
                    loaded_count += 1
                
        except Exception as e:
            # Log warning only; do not crash the program on bad files to ensure flow continuity.
            print(f"❌ Error loading {f_path.name}: {e}")

    if skipped_count > 0:
        print(f"ℹ️  Ignored {skipped_count} items/files (invalid format).")

    # ==================================================
    # Device Logic using InfiniDeviceEnum
    # ==================================================
    # 1. Identify active devices from CLI arguments
    cli_active_devices = []

    # Iterate through the Enum to check corresponding CLI args
    # Logic: Enum name (e.g., CAMBRICON) -> lower() -> arg name (cambricon)
    # Value: InfiniDeviceNames mapping (e.g., "Cambricon")
    for device_enum, device_name in InfiniDeviceNames.items():
        # device_name is like "CPU", "NVIDIA", "Cambricon"
        # arg_name becomes "cpu", "nvidia", "cambricon"
        arg_name = device_name.lower()

        if getattr(args, arg_name, False):
            cli_active_devices.append(device_name)

    print(f"\n[Config Processing]")
    
    for case in cases:
        if "args" not in case or case["args"] is None:
            case["args"] = {}
        case_args = case["args"]

        # 2. Apply Device Overrides (CLI > JSON)
        if cli_active_devices:
            case["device"] = ",".join(cli_active_devices)

        final_dev_str = case.get("device", "").upper()  # Uppercase for easier matching

        # 3. Set Boolean flags in case_args based on final device string
        for device_enum, device_name in InfiniDeviceNames.items():
            arg_name = device_name.lower()
            # Check if the standard name (e.g., "Cambricon" or "NVIDIA") is in the device string
            # We convert both to upper to ensure case-insensitive matching
            is_active = device_name.upper() in final_dev_str
            case_args[arg_name] = is_active

        case_args["save"] = getattr(args, "save", None)
        # Standard arguments (CLI > JSON > Default)
        case_args["bench"] = (
            args.bench if args.bench is not None else case_args.get("bench")
        )

        # Boolean Flags
        case_args["verbose"] = args.verbose or case_args.get("verbose", False)
        case_args["debug"] = args.debug or case_args.get("debug", False)
        case_args["eq_nan"] = args.eq_nan or case_args.get("eq_nan", False)
        case_args["num_prerun"] = (
            args.num_prerun
            if args.num_prerun is not None
            else (case_args.get("num_prerun") or 10)
        )
        case_args["num_iterations"] = (
            args.num_iterations
            if args.num_iterations is not None
            else (case_args.get("num_iterations") or 1000)
        )

    print(f"📂 Processed {len(cases)} cases ready for execution.\n")
    return cases


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
    parser.add_argument(
        "--load",
        nargs="+",
        help="Load test cases from JSON",
    )

    # Default value is None to determine if user provided input
    parser.add_argument("--num_prerun", type=lambda x: max(0, int(x)), default=None)
    parser.add_argument("--num_iterations", type=lambda x: max(0, int(x)), default=None)

    # Add common test arguments (including --save, --bench, etc.)
    add_common_test_args(parser)
    get_hardware_args_group(parser)

    args, unknown_args = parser.parse_known_args()
    # Show what extra arguments will be passed
    if unknown_args:
        print(f"Passing extra arguments to test scripts: {unknown_args}")

    # 1. Discovery
    collector = TestCollector(args.ops_dir)
    if args.list:
        print("Available operators:", collector.get_available_operators())
        return

    # ==========================================================================
    # Branch 1: Load Mode (JSON Data Driven)
    # ==========================================================================
    if args.load:
        # 1. Load and override arguments
        json_cases = load_and_override_cases(args.load, args)
        if not json_cases:
            sys.exit(1)

        # 2. Determine global Bench status (for Summary display)
        bench = json_cases[0]["args"].get("bench")
        verbose = json_cases[0]["args"].get("verbose")

        if verbose:
            print(
                f"Verbose mode: ENABLED (will stop on first error with full traceback)"
            )

        if bench:
            print(f"Benchmark mode: {args.bench.upper()} timing")

        # 3. Initialize and Execute
        test_manager = TestManager(ops_dir=args.ops_dir, verbose=verbose, bench_mode=bench)

        success = test_manager.test(json_cases_list=json_cases)

    # ==========================================================================
    # Branch 2: Local Scan Mode
    # ==========================================================================
    else:
        if args.verbose:
            print(
                f"Verbose mode: ENABLED (will stop on first error with full traceback)"
            )

        if args.bench:
            print(f"Benchmark mode: {args.bench.upper()} timing")

        # 2. Filtering
        target_ops = None
        if args.ops:
            available_ops = set(collector.get_available_operators())
            requested_ops = set(args.ops)
            valid_ops = list(requested_ops & available_ops)
            invalid_ops = list(requested_ops - available_ops)

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

        # 3. Execution Preparation
        # Fill defaults for local mode (since parser default is None)
        global_exec_args = fill_defaults_for_local_mode(args)

        # 4. Initialize API & Execute
        test_manager = TestManager(
            ops_dir=args.ops_dir, verbose=args.verbose, bench_mode=args.bench
        )

        success = test_manager.test(
            target_ops=target_ops, global_exec_args=global_exec_args
        )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
