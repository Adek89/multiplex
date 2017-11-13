import sys
sys.path.append('/home/apopiel/multiplex/MuNeG')
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
import csv

FILE_PATH = "..\\results\\real\\"
with open('..\\results\\functions.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        reader = DanioRerioReader()
        reader.read(row[0])
        homogenity = reader.calcuclate_homogenity()
        real_graph = reader.graph
        classes = [n.label for n in real_graph.nodes()]
        cardinalities = []
        cardinalities.append(len(filter(lambda n : n == 0, classes)))
        cardinalities.append(len(filter(lambda n : n == 1, classes)))
        ev_tools = EvaluationTools()
        baseline_fmeasure = ev_tools.baseline_fmeasure(cardinalities, 'micro')

        with open(FILE_PATH + 'real_baseline.csv' ,'ab') as csvfile:
                    writer = csv.writer(csvfile)

                    writer.writerow([row[0], homogenity])

