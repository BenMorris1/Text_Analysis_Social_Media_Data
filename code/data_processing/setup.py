from setuptools import setup
from Cython.Build import cythonize

setup(
    name='DataProcessing',
    ext_modules=cythonize("data_processing_functions.pyx"),
    zip_safe=False,
)