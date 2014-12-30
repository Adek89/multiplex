from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
#python setup.py build_ext --inplace
setup(
    name = "My hello app",
    include_dirs = [np.get_include()
    ],
    ext_modules = cythonize('*.pyx'), # accepts a glob pattern
)