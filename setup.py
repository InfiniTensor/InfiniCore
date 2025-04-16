from setuptools import setup, find_packages
import os

home_dir = os.path.expanduser("~")

setup(
    name="pyinfini",
    version="0.1.0",
    packages=find_packages(),
    zip_safe=False,
)
