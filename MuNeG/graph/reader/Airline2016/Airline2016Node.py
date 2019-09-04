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

    def __str__(self):
        return self.id.__str__()+" "+self.label.__str__()+" "+self.name.__str__()

    def get_id(self):
        return self.id

    def get_label(self):
        return self.label

    def get_name(self):
        return self.name