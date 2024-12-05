from pathlib import Path

from setuptools import find_packages, setup

# Get requirements from file
with Path.open("requirements.txt", encoding="utf-8") as file_in:
    requirements = [line.strip("\n") for line in file_in]

# Get the library version fomr __version__.py
with Path.open("pysi/__init__.py", encoding="utf-8") as file_in:
    for line_in in file_in:
        line = line_in.split()
        if len(line) < 1:
            continue
        if line[0] == "__version__":
            __version__ = str(line[2]).strip('"').strip("'")

# setup function
setup(
    name="PySi",
    python_requires=">=3.10",
    version=__version__,
    description="A package to make using SIROCCO a wee bit easier.",
    url="https://github.com/saultyevil/pysi",
    author="Edward J. Parkinson",
    author_email="saultyevil@gmail.com",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": ["pysi = pysi.console.cli:cli"]},
)
