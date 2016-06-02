import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="copula_ordinal_regression",
    version="0.0.1",
    author="Robert Walecki",
    author_email="r.walecki14@imperial.ac.uk",
    description=(""),
    license = "BSD",
    keywords = "",
    url = "",
    data_files = [
        ('./copula_ordinal_regression/data/', [
        './copula_ordinal_regression/data/disfa.pklz',
        ])],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    packages=find_packages()
)
