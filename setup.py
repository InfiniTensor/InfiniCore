import glob
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build


class Build(build):
    def run(self):
        subprocess.run(["xmake", "build"])
        subprocess.run(["xmake", "install"])
        subprocess.run(["xmake", "build", "-y", "_infinicore"])
        subprocess.run(["xmake", "install", "_infinicore"])

        installation_dir = os.getenv("INFINI_ROOT", str(Path.home() / ".infini"))
        lib_dir = os.path.join(installation_dir, "lib")
        lib_path = glob.glob(os.path.join(lib_dir, "_infinicore.*"))[0]
        package_dir = os.path.join(self.build_lib, "infinicore")
        os.makedirs(package_dir, exist_ok=True)
        shutil.move(lib_path, package_dir)


setup(package_dir={"": "python"}, cmdclass={"build": Build})
