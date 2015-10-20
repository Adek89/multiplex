__author__ = 'Adrian'

class DanioRerioNode():

    id = 0
    label = 0
    name = ''
    functions = set([])

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return str(self.name)


    @functions.setter
    def functions(self, value):
        self.functions.add(value)

