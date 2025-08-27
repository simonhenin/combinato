#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='combinato',
    version='2.0.0',
    description='Combinato spike sorting package',
    author='Johannes Niediek',
    author_email='jonied@posteo.de',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'matplotlib',
        'PyQt5',
        'joblib',
        'tables',
    ],
    package_data={
        'guisort': ['*.ui', '*.so'],
        'guioverview': ['*.ui'],
    },
    zip_safe=False,
)