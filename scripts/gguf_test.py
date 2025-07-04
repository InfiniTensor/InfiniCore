import os
import sys
import subprocess
from set_env import set_env
import argparse


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

def run(test, device, mode):
    base = os.path.join(PROJECT_DIR, "test", "infiniop-test")
    if mode in ("gguf", "all"):
        if subprocess.run(f"python -m test_generate.testcases.{test}",
                          shell=True, cwd=base).returncode:
            return f"{test}: gguf failed"
    if mode in ("cpp", "all"):
        exe = os.path.join(PROJECT_DIR, "build/linux/x86_64/release/infiniop-test")
        gguf = os.path.join(base, f"{test}.gguf")
        if subprocess.run(f"{exe} {gguf} --{device} --warmup 20 --run 1000",
                          shell=True, cwd=PROJECT_DIR).returncode:
            return f"{test}: cpp failed"
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", nargs="?", default="cpu")
    parser.add_argument("--mode", choices=["gguf", "cpp", "all"], default="all")
    args = parser.parse_args()

    set_env()
    tests = ["add", "gemm", "mul", "random_sample", "swiglu", "clip"]
    fails = [r for t in tests if (r := run(t, args.device, args.mode))]
    
    if not fails:
        print("\n\033[92mAll tests passed!\033[0m")
        sys.exit(0)
    print("\033[91mSome tests failed:\033[0m")
    for f in fails: print(" -", f)
    sys.exit(1)

if __name__ == "__main__":
    main()
