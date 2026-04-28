import os
import subprocess
import platform
import sys
from set_env import (
    set_env,
    set_env_by_config,
)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

def run_cmd(cmd):
    subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True)


def install(xmake_config_flags=""):
    set_env_by_config(xmake_config_flags)
    run_cmd(f"xmake f -y {xmake_config_flags} -cv")
    run_cmd("xmake -y")
    run_cmd("xmake install -y")
    run_cmd("xmake build -y infiniop-test")
    run_cmd("xmake install -y infiniop-test")


if __name__ == "__main__":
    set_env()
    install(" ".join(sys.argv[1:]))
