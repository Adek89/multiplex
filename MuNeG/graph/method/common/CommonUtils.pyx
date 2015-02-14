__author__ = 'Adek'
cimport cython
from random import shuffle
cdef class CommonUtils:

    def __cinit__(self):
        pass

    #cross validation random fold set generator
    def k_fold_cross_validation(self, list items, int k, float percentOfKnownNodes, randomize=False):
        # print ('Percent of known nodes %s' % percentOfKnownNodes)
        cdef float trainignFoldsFloat = k*percentOfKnownNodes
        cdef int trainignFolds = int(trainignFoldsFloat)
        cdef int validationFolds = k-trainignFolds
        # print('Nr of training folds %s:' % trainignFolds)
        # print('Nr of validation folds %s:' % validationFolds)
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
            # print slices
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
            yield training, validation