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
    install_requires=[
        "bson >= 0.5.10",
        "pandas>=2.0.0",
        "numpy>=1.22.0",
        "torch>=2.1.0",
        "transformers>=4.35.0",
        "faiss-cpu>=1.7.0",
        "scikit-learn>=1.2.0",
        "openai>=0.28.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "twine>=3.0"],
    },
    python_requires=">=3.9",
)
