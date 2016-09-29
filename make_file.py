from os import system
import sys

if sys.version_info[0] != 3:
    raise Exception("Should use Python 3 for building extension.")

try:
    system("python3 setup.py build_ext --inplace")
except Exception:

    system("python setup.py build_ext --inplace")