__author__ = 'Adek'

import os.path

if __name__ == '__main__':
    for i in xrange(1,295681):
        if (not os.path.isfile('C:\\Users\\Adek\\PycharmProjects\\MuNeG\\tmp\\output%i.csv' % i)):
            print '%i' % i
