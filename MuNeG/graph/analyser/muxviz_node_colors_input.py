import csv
import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
from graph.reader.StarWars.StarWarsReader import StarWarsReader

reader = StarWarsReader()
reader.read()
realGraph = reader.graph
with open('node_colors.txt','ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
    writer.writerow(['nodeID', 'layerID', 'color', 'size'])
    for layer_id in xrange(1, 7):
        for node in sorted(realGraph.nodes(), key=lambda n : n.id):
            writer.writerow([node.id + 1, layer_id, 'green' if node.label == 0 else 'orange', 10])