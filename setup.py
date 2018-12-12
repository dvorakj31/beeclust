from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name='beeclust',
    packages=find_packages(),
    ext_modules=cythonize('beeclust/beeclust_cython.pyx'),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'NumPy',
    ],
    setup_requires=[
        'Cython',
        'NumPy',
    ],
)
