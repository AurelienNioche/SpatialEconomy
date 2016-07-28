# from distutils.core import setup
# from distutils.extension import Extension
 
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [ 
    Extension("cython_economy", ["cython_economy.pyx"], include_dirs=[np.get_include()])

]

setup(
    name="cython_economy",
    ext_modules = cythonize(extensions),
)
