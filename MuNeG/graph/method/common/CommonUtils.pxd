__author__ = 'Adek'
cimport cython
from random import shuffle
cimport numpy as np
cdef class CommonUtils:
     cpdef prepareFoldClassMat(self, graph, np.ndarray  defaultClassMat, list validation)
     cdef list prepareUnobservdRow(self, int nrOfClasses)