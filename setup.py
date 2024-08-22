from setuptools import setup, find_packages

setup(
    name='DUST3R',
    version='0.1.0',
    # packages=find_packages(),
    packages=['dust3r', 'dust3r.utils', 'dust3r.heads', 'dust3r.datasets', 'dust3r.cloud_opt'],
    author='naver',
)