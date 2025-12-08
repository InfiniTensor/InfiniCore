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
    for test in [
        "acos.py",
        "adaptive_avg_pool1d.py",
        "add.py",
        "addbmm.py",
        "attention.py",
        "causal_softmax.py",
        "clip.py",
        "conv.py",
        #"dequantize_awq.py",
        "floor.py",
        "gelu.py",
        "gemm.py",
        #"layer_norm.py",
        "logsoftmax.py",
        #"lp_norm.py",
        "mul.py",
        "ones.py",
        "random_sample.py",
        "rearrange.py",
        "relu.py",
        "rms_norm.py",
        "rope.py",
        "sigmoid.py",
        #"softmax.py",
        "softplus.py",
        "sub.py",
        "swiglu.py",
        "tanh.py",
        "topkrouter.py",
        "topksoftmax.py",
        "zeros.py",
    ]:
        result = subprocess.run(
            f"python {test} {args} --debug", text=True, encoding="utf-8", shell=True
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    set_env()
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
