#!/usr/bin/env python

import os
from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit iScore/__version__.py
version = {}
with open(os.path.join(here, 'iScore', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()


setup(
    name='iScore',
    version=version['__version__'],
    description="SVM Graph Kernels for Protein Interface Scoring",
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Nicolas Renaud", "Cunliang Geng, Li Xue"],
    author_email='n.renaud@esciencecenter.nl',
    url='https://github.com/DeepRank/iScore',
    packages=find_packages(),
    package_dir={'iScore': 'iScore'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='iScore',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'intended audience :: science/research',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Biochemistry'
    ],
    test_suite='tests',

    install_requires=[ 'numpy >= 1.13', 'scipy', 'biopython', 
                       'mpi4py', 'h5py','matplotlib','libsvm'],


    extras_require={
        'dev': ['prospector[with_pyroma]', 'yapf', 'isort'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test': ['nose','coverage', 'pycodestyle', 'pytest', 'pytest-cov', 'pytest-runner','coveralls'],
    }
)