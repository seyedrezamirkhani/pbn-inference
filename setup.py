import pathlib

from setuptools import find_packages, setup

CURRENT_PATH = pathlib.Path(__file__).parent
README = (CURRENT_PATH / "README.md").read_text()

setup(
    name="pbn_inference",
    version="1.0.0",
    description="A package with methods for Probabilistic Boolean Network inference.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/UoS-PLCCN/pbn_inference/",
    author="Evangelos Chatzaroulas",
    author_email="e.chatzaroulas@surrey.ac.uk",
    license="MIT",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    package_data={"pbn_inference.bittner": ["data/*.xls", "data/*.pkl"]},
    install_requires=[
        "networkx",
        "numpy",
        "pandas",
        "xlrd",
        "scipy",
        "sklearn",
        "numba",
        "tqdm",
        "psutil",
    ],
    extras_require={
        "dev": ["pytest", "black", "rope", "wandb"],
    },
    python_requires=">3.7",
)
