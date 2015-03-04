__author__ = 'Adek'
from graph.method.ensamble.EnsambleLearning import EnsambleLearning
import graph.method.ica.ClassifiersUtil as cu
import graph.gen.GraphGenerator as gg

if __name__ == '__main__':
    generator = gg.GraphGenerator(10, 2, [1, 2], 5, 5, 1, ["L1", "L2"])
    ensambleLearn = EnsambleLearning(generator.generate(), 5, 2)
    ensambleLearn.ensamble()