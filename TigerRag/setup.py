from pathlib import Path
from setuptools import find_packages, setup

with open(Path(__file__).absolute().parents[0] / "tigerrag" / "VERSION") as _f:
    __version__ = _f.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tigerrag",
    version=__version__,
    description="TigerRAG SDK to utilize and combine retrieval and llm using external datasets.",
    packages=find_packages(),
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
    include_package_data=True,
    package_data={
        "tigerrag": ["demo/movie_recs/*.csv"]
    },
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=6.0", "twine>=3.0"],
    },
    python_requires=">=3.9",
)
