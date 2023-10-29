from pathlib import Path
from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "tigertune" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tigertune",
    version=__version__,
    description="TigerTune SDK to fine tune a model with training data.",
    package_dir={"": "tigertune"},
    packages=find_packages(where="tigertune"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tigerlab-ai/tiger",
    author="TigerLab",
    author_email="tigerlab.ai@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=6.0", "twine>=3.0"],
    },
    python_requires=">=3.9",
)
