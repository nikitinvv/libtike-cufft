from skbuild import setup
from setuptools import find_packages

setup(
    name='ptychocg',
    author='Viktor Nikitin',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages('src'),
    zip_safe=False,
)
