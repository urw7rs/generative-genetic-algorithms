from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gga",
    version="0.0.0",
    description="Generative Genetic Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chanhyuk Jung",
    package_dir={"": "gga"},
    packages=find_packages(where="gga"),
    python_requires=">=3.7",
    install_requires=["flax", "ciclo", "tensorflow_datasets", "jsonargparse"],
    extras_require={
        "test": ["pytest", "chex"],
        "dev": ["black", "flake8", "bumpver"],
    },
)
