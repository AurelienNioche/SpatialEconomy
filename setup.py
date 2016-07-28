# from distutils.core import setup
# from distutils.extension import Extension
 
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [ 
    Extension("eco.c_economy", ["eco/c_economy.pyx"], include_dirs=[np.get_include()])

]

setup(
    name="c_economy",
    ext_modules = cythonize(extensions),
)
