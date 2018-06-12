#from distutils.core import setup
from setuptools import setup

setup(
    name='iScore',
    description='Graph based Scoring of protein-protein conformations',
    version='0.0',
    #url='https://github.com/iScore',
    packages=['iScore'],


    install_requires=[
        'numpy >= 1.13',
        'scipy'],

    extras_require= {
        'test': ['nose', 'coverage', 'pytest',
                 'pytest-cov','codacy-coverage','coveralls'],
    }
)