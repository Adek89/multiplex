__author__ = 'Adek'

class Edge:

    node1 = None
    node2 = None
    data = None

    def __init__(self, node1, node2, data):
        self.node1 = node1
        self.node2 = node2
        self.data = data