try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name = "My hello app",
    ext_modules = cythonize('*.pyx') # accepts a glob pattern
)