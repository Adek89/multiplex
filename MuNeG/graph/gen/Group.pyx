'''
Created on 6 lut 2014

@author: Adek
'''
cimport cython
cdef class Group:


    def __cinit__(self, str types, int number):
        '''
        Constructor 
        '''
        self.gType = types
        self.gNumber = number


    property gType:
        def __get__(self):
          return self.gType
        def __set__(self, str value):
          self.gType = value