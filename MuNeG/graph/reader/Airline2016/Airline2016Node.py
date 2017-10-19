class Airline2016Node():

    id = 0
    label = None
    name = None

    def __init__(self, id, label, name):
        self.id = id
        self.label = label
        self.name = name

    def __repr__(self):
        return str(self.name)