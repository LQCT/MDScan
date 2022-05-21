import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The txt of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="mdscan",
    version="0.0.2",
    description="RMSD-Based HDBSCAN Clustering of Long Molecular Dynamics",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/LQCT/MDScan.git",
    author="Roy González-Alemán",
    author_email="roy_gonzalez@fq.uh.cu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(where='src'),
    package_dir={"":"src"},
    include_package_data=True,
    install_requires=['numpy==1.21 ', 'numba>=0.55', 'mdtraj>=1.9.5',
                      'numpy-indexed>=0.3', 'pandas>=1.4'],
    entry_points={
        "console_scripts": [
            "mdscan=src.mdscan:main",
        ]
    },
)
