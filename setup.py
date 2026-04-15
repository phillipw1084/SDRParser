from setuptools import setup, find_packages

setup(
    name="sdrparser",
    version="0.1.0",
    description="SDR++ audio stream parser for DMR, NXDN, and P25 digital voice formats",
    author="SDRParser Contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "sdrparser=sdrparser.main:main",
        ],
    },
)
