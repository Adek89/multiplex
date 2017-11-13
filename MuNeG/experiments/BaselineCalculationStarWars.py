import sys
sys.path.append('/home/apopiel/multiplex/MuNeG')
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.reader.StarWars.StarWarsReader import StarWarsReader
import csv

FILE_PATH = "..\\results\\real_sw\\"
options = ["default", "anakin_light", "anakin_dark"]
reader = StarWarsReader()
reader.read(isAnakinEqualVader=True)
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

            writer.writerow(["anakin_dark", homogenity])

