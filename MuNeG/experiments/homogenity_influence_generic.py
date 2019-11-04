import sys
sys.path.append('D:\\pycharm_workspace\\multiplex\\MuNeG\\')
sys.path.append('/home/apopiel/multiplex/MuNeG/')

from experiments.DecisionFusionGeneric import DecisionFusion
import sklearn.metrics as metrics
import csv
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.reader.Cora.CoraNode import CoraNode
from graph.reader.HospitalWard.HospitalWardNode import HospitalWardNode
from graph.reader.HighSchool.HighSchoolNode import HighSchoolNode
from graph.reader.Airline2016.Airline2016Node import Airline2016Node
import networkx as nx
import math
import pickle

def node_destringizer(value):
    value_str = str(value)
    value_splitted = value_str.split()
    if reader == 'Cora':
        return CoraNode(int(value_splitted[0]),int(value_splitted[1]))
    elif reader == 'HospitalWard':
        return HospitalWardNode(int(value_splitted[0]),int(value_splitted[1]))
    elif reader == 'HighSchool':
        return HighSchoolNode(int(value_splitted[0]),int(value_splitted[1]))
    elif reader == 'Airline':
        return Airline2016Node(int(value_splitted[0]),int(value_splitted[1]),value_splitted[2])

fold = int(sys.argv[1])
r = int(sys.argv[2])
homogenity = float(sys.argv[3])
reader = sys.argv[4]
aucs = {}
ev = EvaluationTools()
df = DecisionFusion(math.fabs(fold))
graph = nx.read_gml("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/graph_" + str(r) + "_" + str(homogenity) + ".gml", destringizer=node_destringizer)
df.realGraph = graph
folds_file = open("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/folds" + str(fold) + "_" + str(r) + ".tmp", "rb")
df.folds = pickle.load(folds_file)
folds_file.close()

stopCondition = True
i = 0
realGraphClassMat, realNrOfClasses = df.nu.createClassMat(df.realGraph)
df.realGraphClassMat = realGraphClassMat
df.realNrOfClasses = realNrOfClasses
df.flatLBP()
df.evaluation()
aucs_in_iteration = []
roc_methods = [str(c_id) for c_id in xrange(0,realNrOfClasses)]
roc_methods.append("micro")
roc_methods.append("macro")
for key in roc_methods:
    roc_auc = metrics.auc(df.fprs_per_method[key], df.tprs_per_method[key])
    aucs_in_iteration.append(roc_auc)
aucs.update({i:aucs_in_iteration})
homogenity_distribution, node_ids = df.calculate_homogenity(df.realGraph)
avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
accuracy = ev.calculateAccuracy(df.realLabels, df.realFlatResult)
with open("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([avg_homogenity, aucs_in_iteration, accuracy, fold, nx.density(graph), r, df.homogenities_during_experiment ])
i = i + 1