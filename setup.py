#from distutils.core import setup
from setuptools import setup

setup(
    name='iScore',
    description='SVM Graph Kernels for Protein Interface Scoring',
    version='0.0',
    url='https://github.com/DeepRank/iScore',
    packages=['iScore'],


    install_requires=[
        'numpy >= 1.13',
        'scipy'],

    extras_require= {
        'test': ['nose', 'coverage', 'pytest',
                 'pytest-cov','codacy-coverage','coveralls'],
    }
)
