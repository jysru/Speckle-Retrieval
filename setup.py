from pathlib import Path
import setuptools

with (Path(__file__).parent / "README.md").open(encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="speckleret",
    version="0.1",
    author="Jysru",
    author_email="jysru@pm.me",
    description="Speckle Phase Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jysru/Speckle-Retrieval",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy", "scipy", "matplotlib", "torch>=2.0", "scikit-image"],
    extras_require={"dev": ["pytest>=7"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)