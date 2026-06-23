"""Setup de instalacion para el proyecto RADAR Cibest."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
REQUIREMENTS_FILE = ROOT / "requirements.txt"
requirements = []
if REQUIREMENTS_FILE.exists():
    requirements = [
        line.strip() for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="radar_cibest",
    version="1.0.0",
    description="Sistema analitico multicriterio hibrido para internacionalizacion de Grupo Cibest",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Jhon Adarve - Direccion de Estrategia",
    author_email="jadarve@bancolombia.com.co",
    url="",  # repositorio interno de Cibest
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=("tests", "tests.*", "notebooks")),
    #packages=find_packages(exclude=("tests", "tests.*", "notebooks")),
    include_package_data=True,
    python_requires=">=3.9,<3.13",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "radar-extract=src.data_extraction.pipeline:main",
        ],
    },
)
