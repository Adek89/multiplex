try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize

#python setup.py build_ext --inplace
setup(
    name = "My hello app",
    include_dirs = [np.get_include()
    ],
    ext_modules = cythonize('*.pyx'), # accepts a glob pattern
)