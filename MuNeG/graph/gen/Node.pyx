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

    def __cinit__(self, g.Group group, int label, int id):
        '''
        Constructor
        '''
        self.group = group
        self.label = label
        self.id = id
        
    def __str__(self):
        return self.id.__str__()+" "+self.label.__str__()+" "+self.group.__str__()
    
    def get_id(self):
        return self.id
    
    def get_label(self):
        return self.label
    
    def get_group(self):
        return self.group

    def __reduce__(self):
        # a tuple as specified in the pickle docs - (class_or_constructor,
        # (tuple, of, args, to, constructor))
        return (self.__class__, (self.group, self.label, self.id))

    property group:
        def __get__(self):
          return self.group
        def __set__(self, g.Group value):
          self.group = value

    property label:
        def __get__(self):
          return self.label
        def __set__(self, int value):
          self.label = value

    property id:
        def __get__(self):
          return self.id
        def __set__(self, int value):
          self.id = value