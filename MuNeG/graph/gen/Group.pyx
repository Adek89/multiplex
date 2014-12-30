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


    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor,
        # (tuple, of, args, to, constructor))
        return (self.__class__, (self.gType, self.gNumber))

    property gType:
        def __get__(self):
          return self.gType
        def __set__(self, str value):
          self.gType = value