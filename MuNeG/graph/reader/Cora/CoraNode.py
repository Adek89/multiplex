class CoraNode:
    id = 0
    label = ''

    def __init__(self, id, status=''):
        self.id = id
        self.label = status

    def __str__(self):
        return self.id.__str__()+" "+self.label.__str__()