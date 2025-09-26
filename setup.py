import glob
import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

INSTALLATION_DIR = os.getenv("INFINI_ROOT", str(Path.home() / ".infini"))

LIB_DIR = os.path.join(INSTALLATION_DIR, "lib")

PACKAGE_NAME = "infinicore"

PACKAGE_DIR = os.path.join(INSTALLATION_DIR, PACKAGE_NAME)


class BuildPy(build_py):
    def run(self):
        subprocess.run(["xmake", "build"])
        subprocess.run(["xmake", "install"])
        subprocess.run(["xmake", "build", "-y", "infinicore"])
        subprocess.run(["xmake", "install", "infinicore"])
        built_lib = glob.glob(os.path.join(LIB_DIR, f"{PACKAGE_NAME}.*"))[0]
        os.makedirs(PACKAGE_DIR, exist_ok=True)
        self.copy_file(built_lib, PACKAGE_DIR)


setup(
    cmdclass={"build_py": BuildPy},
    package_dir={"": "."},
)
