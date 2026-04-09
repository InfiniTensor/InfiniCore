"""Sync operator implementations from InfiniOps into InfiniCore.

This script invokes InfiniOps's code generator to produce legacy C API wrappers,
then copies the generated headers and source files into InfiniCore's tree,
replacing the hand-written operator dispatch files.

Usage:
    python scripts/sync_infiniops.py /path/to/InfiniOps [--devices cpu nvidia ...]
    python scripts/sync_infiniops.py /path/to/InfiniOps --ops gemm add
    python scripts/sync_infiniops.py /path/to/InfiniOps --dry-run
"""

import argparse
import difflib
import pathlib
import shutil
import subprocess
import sys

INFINICORE_ROOT = pathlib.Path(__file__).resolve().parent.parent
INFINICORE_INCLUDE_OPS = INFINICORE_ROOT / "include" / "infiniop" / "ops"
INFINICORE_SRC_OPS = INFINICORE_ROOT / "src" / "infiniop" / "ops"


def run_generator(infiniops_root, devices):
    """Run InfiniOps's `generate_wrappers.py` and return the generated directory."""
    generator = infiniops_root / "scripts" / "generate_wrappers.py"

    if not generator.exists():
        print(f"Error: generator not found at {generator}", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, str(generator), "--devices"] + devices
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=infiniops_root, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: generator failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    generated_dir = infiniops_root / "generated"

    if not generated_dir.exists():
        print(f"Error: expected output directory {generated_dir} not found", file=sys.stderr)
        sys.exit(1)

    return generated_dir


def discover_generated_ops(generated_dir):
    """Return a sorted list of operator names that were generated."""
    include_dir = generated_dir / "include"

    return sorted(header.stem for header in include_dir.glob("*.h"))


def show_diff(old_path, new_content, label):
    """Show a unified diff between an existing file and new content."""
    if old_path.exists():
        old_lines = old_path.read_text().splitlines(keepends=True)
    else:
        old_lines = []

    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{label}", tofile=f"b/{label}"
    )
    diff_str = "".join(diff)

    if diff_str:
        print(diff_str)

    return bool(diff_str)


def sync_operator(op_name, generated_dir, dry_run=False, verbose=False):
    """Copy generated files for one operator into InfiniCore."""
    gen_header = generated_dir / "include" / f"{op_name}.h"
    gen_source = generated_dir / "src" / op_name / "operator.cc"
    dst_header = INFINICORE_INCLUDE_OPS / f"{op_name}.h"
    dst_source = INFINICORE_SRC_OPS / op_name / "operator.cc"

    if not gen_header.exists():
        print(f"  Warning: generated header not found: {gen_header}", file=sys.stderr)

        return False

    if not gen_source.exists():
        print(f"  Warning: generated source not found: {gen_source}", file=sys.stderr)

        return False

    new_header = gen_header.read_text()
    new_source = gen_source.read_text()
    header_changed = False
    source_changed = False

    if verbose or dry_run:
        header_label = f"include/infiniop/ops/{op_name}.h"
        source_label = f"src/infiniop/ops/{op_name}/operator.cc"
        header_changed = show_diff(dst_header, new_header, header_label)
        source_changed = show_diff(dst_source, new_source, source_label)

    if dry_run:
        if not header_changed and not source_changed:
            print(f"  {op_name}: no changes")

        return header_changed or source_changed

    # Ensure destination directories exist.
    dst_header.parent.mkdir(parents=True, exist_ok=True)
    dst_source.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(gen_header, dst_header)
    shutil.copy2(gen_source, dst_source)
    print(f"  {op_name}: synced")

    return True


def verify_compilation(op_name, infiniops_root):
    """Syntax-check the replaced `operator.cc` compiles with the right include paths."""
    source = INFINICORE_SRC_OPS / op_name / "operator.cc"
    cmd = [
        "g++", "-std=c++17", "-fsyntax-only",
        f"-I{INFINICORE_ROOT / 'include'}",
        f"-I{infiniops_root / 'src'}",
        str(source),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  {op_name}: COMPILE FAILED", file=sys.stderr)

        if result.stderr:
            # Show the first few lines of the error.
            for line in result.stderr.splitlines()[:10]:
                print(f"    {line}", file=sys.stderr)

        return False

    print(f"  {op_name}: compile OK")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync InfiniOps operator wrappers into InfiniCore.",
    )
    parser.add_argument(
        "infiniops_path",
        type=pathlib.Path,
        help="Path to the InfiniOps project root.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu"],
        help="Devices to generate for (default: cpu).",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=None,
        help="Only sync specific operators (default: all generated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show diffs without modifying files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Syntax-check each replaced file after syncing.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show diffs even when not in dry-run mode.",
    )
    args = parser.parse_args()

    infiniops_root = args.infiniops_path.resolve()

    if not (infiniops_root / "scripts" / "generate_wrappers.py").exists():
        print(
            f"Error: {infiniops_root} does not look like an InfiniOps project.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: Run InfiniOps generator.
    print("=== Generating wrappers ===")
    generated_dir = run_generator(infiniops_root, args.devices)

    # Step 2: Discover what was generated.
    all_ops = discover_generated_ops(generated_dir)
    ops_to_sync = args.ops if args.ops else all_ops

    skipped = [op for op in ops_to_sync if op not in all_ops]

    if skipped:
        print(
            f"Warning: requested ops not found in generated output: {skipped}",
            file=sys.stderr,
        )
        ops_to_sync = [op for op in ops_to_sync if op in all_ops]

    if not ops_to_sync:
        print("Nothing to sync.")

        return

    # Step 3: Sync files.
    action = "Previewing" if args.dry_run else "Syncing"
    print(f"\n=== {action} {len(ops_to_sync)} operator(s): {', '.join(ops_to_sync)} ===")

    synced = []

    for op_name in ops_to_sync:
        changed = sync_operator(
            op_name, generated_dir, dry_run=args.dry_run, verbose=args.verbose,
        )

        if changed and not args.dry_run:
            synced.append(op_name)

    if args.dry_run:
        return

    # Step 4: Verify compilation.
    if args.verify and synced:
        print("\n=== Verifying compilation ===")
        failures = []

        for op_name in synced:
            if not verify_compilation(op_name, infiniops_root):
                failures.append(op_name)

        if failures:
            print(
                f"\nCompilation failed for: {', '.join(failures)}", file=sys.stderr,
            )
            sys.exit(1)

    print(f"\nDone. Synced {len(synced)} operator(s).")


if __name__ == "__main__":
    main()
