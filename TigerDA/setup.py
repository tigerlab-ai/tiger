from pathlib import Path
from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "tigerda" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tigerda",
    version=__version__,
    description="tigerda SDK for data augmentation.",
    package_dir={"": "tigerda"},
    packages=find_packages(where="tigerda"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tigerlab-ai/tiger",
    author="TigerLab",
    author_email="tigerlab.ai@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "transformers",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "twine>=3.0"],
    },
    python_requires=">=3.9",
)
