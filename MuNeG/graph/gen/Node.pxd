'''
Created on 6 lut 2014

@author: Adek
'''
cimport cython
cimport graph.gen.Group as g
cdef class Node:
    '''
    classdocs
    '''
    cdef g.Group group
    cdef int label
    cdef int id