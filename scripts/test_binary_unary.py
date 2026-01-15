import os
import subprocess
from set_env import set_env
import sys

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test", "infiniop")
)
os.chdir(PROJECT_DIR)


def run_tests(args):
    failed = []
    
    # Binary operators (重构过的)
    binary_tests = [
        "div.py",
        "pow.py",
        "mod.py",
        "min.py",
        "max.py",
    ]
    
    # Unary operators (重构过的)
    unary_tests = [
        "abs.py",
        "log.py",
        "cos.py",
        "sqrt.py",
        "neg.py",
        "sign.py",
        "reciprocal.py",
        "round.py",
        "floor.py",
        "ceil.py",
        "erf.py",
        "cosh.py",
        "sinh.py",
        "tan.py",
        "acos.py",
        "acosh.py",
        "asin.py",
        "asinh.py",
        "atan.py",
        "atanh.py",
    ]
    
    all_tests = binary_tests + unary_tests
    
    print("\033[94m" + "=" * 60 + "\033[0m")
    print("\033[94mTesting Binary and Unary Operators (Refactored)\033[0m")
    print("\033[94m" + "=" * 60 + "\033[0m")
    print(f"\033[94mTotal tests: {len(all_tests)}\033[0m")
    print(f"\033[94m  - Binary operators: {len(binary_tests)}\033[0m")
    print(f"\033[94m  - Unary operators: {len(unary_tests)}\033[0m")
    print()
    
    for test in all_tests:
        if not os.path.exists(test):
            print(f"\033[93m[SKIP] {test} - test file not found\033[0m")
            continue
            
        print(f"\033[96m[RUN] {test}\033[0m", end=" ... ", flush=True)
        result = subprocess.run(
            f"python3 {test} {args}", 
            text=True, 
            encoding="utf-8", 
            shell=True,
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"\033[91m[FAIL]\033[0m")
            print(f"\033[91mError output:\033[0m")
            print(result.stderr)
            failed.append(test)
        else:
            print(f"\033[92m[PASS]\033[0m")
    
    return failed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test refactored binary and unary operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on CPU only (default)
  python3 scripts/test_binary_unary.py --cpu
  
  # Test on NVIDIA GPU only
  python3 scripts/test_binary_unary.py --nvidia
  
  # Test on both CPU and NVIDIA
  python3 scripts/test_binary_unary.py --cpu --nvidia
  
  # Test with debug mode
  python3 scripts/test_binary_unary.py --cpu --debug
  
  # Test with profiling
  python3 scripts/test_binary_unary.py --nvidia --profile
        """
    )
    
    # Device selection arguments (same as test files)
    parser.add_argument("--cpu", action="store_true", help="Run CPU tests")
    parser.add_argument("--nvidia", action="store_true", help="Run NVIDIA GPU tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args, unknown = parser.parse_known_args()
    
    # Build command line arguments to pass to test files
    test_args = []
    if args.cpu:
        test_args.append("--cpu")
    if args.nvidia:
        test_args.append("--nvidia")
    if args.debug:
        test_args.append("--debug")
    if args.profile:
        test_args.append("--profile")
    
    # Add any unknown arguments (for compatibility)
    test_args.extend(unknown)
    
    set_env()
    failed = run_tests(" ".join(test_args))
    
    print()
    print("\033[94m" + "=" * 60 + "\033[0m")
    if len(failed) == 0:
        print("\033[92m✓ All tests passed!\033[0m")
    else:
        print(f"\033[91m✗ {len(failed)} test(s) failed:\033[0m")
        for test in failed:
            print(f"\033[91m  - {test}\033[0m")
    print("\033[94m" + "=" * 60 + "\033[0m")
    
    exit(len(failed))
