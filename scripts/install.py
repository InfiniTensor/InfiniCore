import os
import subprocess
import platform
import sys
from set_env import (
    set_env,
    ensure_metax_hpc_compiler_includes,
    xmake_flags_need_metax_aten_torch_includes,
)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

def run_cmd(cmd):
    subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True)


def install(xmake_config_flags=""):
    if xmake_flags_need_metax_aten_torch_includes(xmake_config_flags):
        ensure_metax_hpc_compiler_includes()
    run_cmd(f"xmake f {xmake_config_flags} -cv")
    run_cmd("xmake")
    run_cmd("xmake install")
    run_cmd("xmake build infiniop-test")
    run_cmd("xmake install infiniop-test")


if __name__ == "__main__":
    set_env()
    install(" ".join(sys.argv[1:]))
