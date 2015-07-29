__author__ = 'Adek'
from graph.method.ensamble.EnsambleClassification import EnsambleClassification
import graph.gen.GraphGenerator as gg
import graph.method.ica.ClassifiersUtil as cu
if __name__ == '__main__':
    generator = gg.GraphGenerator(10, 2, [1, 2], 5, 5, 1, ["L1", "L2"])
    ensambleClass = EnsambleClassification(cu.knownModels(), generator.generate(), 0.2)
    ensambleClass.classify()