import sys
import argparse
from framework import get_hardware_args_group, add_common_test_args
# 导入我们新抽取的 API
from framework.api import TestRunner, TestDiscoverer

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
    # 1. Setup CLI
    parser = argparse.ArgumentParser(
        description="Run InfiniCore operator tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_epilog()
    )
    
    parser.add_argument("--ops-dir", type=str, help="Path to ops directory")
    parser.add_argument("--ops", nargs="+", help="Run specific operators")
    parser.add_argument("--list", action="store_true", help="List tests")
    
    add_common_test_args(parser)
    get_hardware_args_group(parser)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Passing extra args: {unknown}")

    # 2. Init API Runner
    runner = TestRunner(
        ops_dir=args.ops_dir, 
        verbose=args.verbose, 
        bench_mode=args.bench
    )

    # 3. Handle --list command
    if args.list:
        print("Available operators:", runner.list_operators())
        return

    # 4. Filter logic (Validating user input) 
    # This logic belongs in the CLI layer because it's about User Input Validation
    target_ops = None
    if args.ops:
        available = set(runner.list_operators())
        requested = set(args.ops)
        
        valid = list(requested & available)
        invalid = list(requested - available)
        
        if invalid:
            print(f"⚠️  Warning: Operators not found: {', '.join(invalid)}")
        
        if not valid:
            print(f"⚠️  No valid operators found. Running ALL tests...")
        else:
            print(f"🎯 Targeted operators: {', '.join(valid)}")
            target_ops = valid

    if args.bench:
        print(f"Benchmark mode: {args.bench.upper()} timing")

    # 5. Execute via API
    success = runner.run(target_ops)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
