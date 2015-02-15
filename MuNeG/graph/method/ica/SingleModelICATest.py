__author__ = 'Adek'
from graph.method.ica.SingleModelICA import SingleModelICA
import graph.method.ica.ClassifiersUtil as cu
import graph.gen.GraphGenerator as gg
if __name__ == '__main__':
    generator = gg.GraphGenerator(10, 2, [1, 2], 5, 5, 1, ["L1", "L2"])
    singleModel = SingleModelICA(generator.generate(), 0.5, 2, cu.giveNaiveBayes())
    singleModel.classify()