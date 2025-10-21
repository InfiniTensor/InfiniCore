import subprocess

from setuptools import setup
from setuptools.command.build import build


class Build(build):
    def run(self):
        subprocess.run(["xmake", "build"])
        subprocess.run(["xmake", "install"])
        subprocess.run(["xmake", "build", "-y", "_infinicore"])
        subprocess.run(["xmake", "install", "_infinicore"])


setup(package_dir={"": "python"}, cmdclass={"build": Build})
