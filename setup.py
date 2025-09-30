import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

INSTALLATION_DIR = os.getenv("INFINI_ROOT", str(Path.home() / ".infini"))

LIB_DIR = os.path.join(INSTALLATION_DIR, "lib")

PYTHON_PACKAGE_DIR = os.path.join("python", "infinicore")

LIB_PACKAGE_DIR = os.path.join(LIB_DIR, "infinicore")


class BuildPy(build_py):
    def run(self):
        subprocess.run(["xmake", "build"])
        subprocess.run(["xmake", "install"])
        subprocess.run(["xmake", "build", "-y", "infinicore"])
        subprocess.run(["xmake", "install", "infinicore"])

        if os.path.exists(LIB_PACKAGE_DIR):
            shutil.rmtree(LIB_PACKAGE_DIR)

        shutil.copytree(PYTHON_PACKAGE_DIR, LIB_PACKAGE_DIR)


setup(cmdclass={"build_py": BuildPy})
