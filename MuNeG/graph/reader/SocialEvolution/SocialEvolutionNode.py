class SocialEvolutionNode():

    id = 0
    label = None

    def __init__(self, id, label):
        self.id = id
        self.label = label

    def __repr__(self):
        return str(self.id)