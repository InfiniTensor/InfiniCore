import subprocess
from setuptools import setup, find_packages
from setuptools.command.build import build

def run_xmake_build():
    print("Running xmake build...")
    subprocess.run(["xmake", "build"], check=True)
    subprocess.run(["xmake", "install"], check=True)
    subprocess.run(["xmake", "build", "-y", "_infinicore"], check=True)
    subprocess.run(["xmake", "install", "_infinicore"], check=True)

class Build(build):
    def run(self):
        run_xmake_build()
        super().run()

setup(
    # 1. Find main packages and manually add test/framework packages
    packages=find_packages(where="python") + [
        "infinicore.test", 
        "infinicore.test.framework"
    ],
    
    # 2. Directory mappings
    package_dir={
        "": "python",  # Root package is under python/ directory
        "infinicore.test": "test/infinicore"  # Intermediate package mapping
    },
    
    # 3. Register commands
    cmdclass={
        "build": Build
    }
)
