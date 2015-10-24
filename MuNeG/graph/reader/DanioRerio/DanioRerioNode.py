__author__ = 'Adrian'

class DanioRerioNode():

    id = 0
    label = 0
    name = ''

    def __init__(self, id):
        self.id = id
        self.functions = set([])

    def __repr__(self):
        return str(self.name)

    @property
    def functions(self):
        return self.functions


    @functions.setter
    def functions(self, value):
        self.functions.add(value)

