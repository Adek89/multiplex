import sys
sys.path.append('/home/apopiel/multiplex/MuNeG')
from graph.evaluation.EvaluationTools import EvaluationTools
import graph.reader.syntethic.MuNeGGraphReader as reader
import math
import csv

def build_file_name(NUMBER_OF_NODES, AVERAGE_GROUP_SIZE, GROUP_LABEL_HOMOGENITY, PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS):
    homogenity = GROUP_LABEL_HOMOGENITY if GROUP_LABEL_HOMOGENITY in [5.5, 6.5, 7.5, 8.5, 9.5] else int(math.floor(GROUP_LABEL_HOMOGENITY))
    group = int(PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP )
    between_other_groups = PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS if PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS in [0.1, 0.5] else int(PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS)
    nodes = int(round(float(NUMBER_OF_NODES) / 100.0) * 100)
    return 'muneg_' + str(nodes) + '_' + str(AVERAGE_GROUP_SIZE) + '_' + str(
        homogenity) + '_' + str(group) + '_' + str(
        between_other_groups) + '_' + str(nrOfLayers) + '.gml'

FILE_PATH = "..\\results\\synthetic\\"
nodes = int(sys.argv[1])
size = int(sys.argv[2])
for label in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
    for probIn in [5, 6, 7, 8, 9]:
        for probBetween in [0.1, 0.5, 1, 2, 3, 4, 5]:
            for nrOfLayers in [2, 3, 4, 5, 6, 8, 10, 21]:
                synthetic = reader.read_from_gml('..\\results', build_file_name(nodes, size, label, probIn, probBetween),)
                real_homogenity = reader.calcuclate_homogenity(synthetic)
                classes = [n.label for n in synthetic.nodes()]
                cardinalities = []
                cardinalities.append(len(filter(lambda n : n == 0, classes)))
                cardinalities.append(len(filter(lambda n : n == 1, classes)))
                ev_tools = EvaluationTools()
                baseline_fmeasure = ev_tools.baseline_fmeasure(cardinalities, 'micro')

                with open(FILE_PATH + 'synt_baseline_' + str(nodes) + '_'+ str(size) + '.csv' ,'ab') as csvfile:
                    writer = csv.writer(csvfile)

                    writer.writerow([nodes, size, label,
                              probIn, probBetween,
                                nrOfLayers, baseline_fmeasure])

