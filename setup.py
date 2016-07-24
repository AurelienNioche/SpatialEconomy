from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy 

extensions = [ 
    Extension("writer_main_wisdom_of_crowds", ["writer_main_wisdom_of_crowds.pyx"])

]

setup(
    cmdclass = { 'buid_ext':build_ext },
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
