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

    def __str__(self):
        return self.gType.__str__() + ' ' + str(self.gNumber)


    property gType:
        def __get__(self):
          return self.gType
        def __set__(self, str value):
          self.gType = value

    property gNumber:
        def __get__(self):
          return self.gNumber
        def __set__(self, int value):
          self.gNumber = value