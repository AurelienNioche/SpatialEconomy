# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [ 
    Extension("c_writer_main_wisdom_of_crowds", ["c_writer_main_wisdom_of_crowds.pyx"], include_dirs=[np.get_include()])

]

setup(
    name="c_writer_main_wisdom_of_crowds",
    ext_modules = cythonize(extensions),
)
