from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="rag-experiment-accelerator",
    version="0.9",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9, <4",
)
