__author__ = 'Adek'
cimport cython
from random import shuffle
import networkx as nx
cdef class CommonUtils:

    def __cinit__(self):
        pass

    #cross validation random fold set generator
    def k_fold_cross_validation(self, list items, int k, randomize=False):
        print ('Folds %s' % k)
        cdef int trainignFolds = int(k-1)
        cdef int validationFolds = k-trainignFolds
        print('Nr of training folds %s:' % trainignFolds)
        print('Nr of validation folds %s:' % validationFolds)
        if randomize:
            items = list(items)
            shuffle(items)
        cdef list slices
        cdef int i
        cdef list validation = []
        cdef list training
        cdef list s
        cdef int item
        cdef int j
        cdef int index
        for i in xrange(k):
            slices = [items[x::k] for x in xrange(k)]
            print slices
            print 'slices: ' + str(slices)
            validation = []
            training = []
            for j in xrange(k):
                index = i + j
                if (index >= k):
                    index -= k
                if (j < validationFolds):
                    if (validation.__len__() == 0):
                        validation = slices[index]
                    else:
                        validation += slices[index]
                    # print ('Index %i added as validation ' % index)
                else:
                    if (training.__len__() == 0):
                        training = slices[index]
                    else:
                        training += slices[index]
                    # print ('Index %i added as training ' % index)
            training = sorted(training)
            validation = sorted(validation)
            print 'training length: ' + str(training.__len__())
            print 'validation length: ' + str(validation.__len__())
            yield training, validation


    cpdef prepareFoldClassMat(self, graph, np.ndarray  defaultClassMat, list validation):
        cdef classMat = defaultClassMat.copy()
        cdef int nrOfClasses = classMat.shape[1]
        cdef int i
        cdef list row
        cdef list sortedNodes
        cdef adjMat
        for i in range(0, defaultClassMat.__len__()):
            if i in validation:
                row = self.prepareUnobservdRow(nrOfClasses)
                classMat[i] = row

        sortedNodes = sorted(graph.nodes())
        adjMat = nx.adjacency_matrix(graph, sortedNodes)

        return classMat, adjMat, sortedNodes

    cdef list prepareUnobservdRow(self, int nrOfClasses):
        cdef list row = []
        cdef int i
        for i in range(0, nrOfClasses):
            row.append(0.5)
        return row