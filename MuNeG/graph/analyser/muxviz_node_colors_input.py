import csv

from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader

reader = DanioRerioReader()
reader.read('GO:0005634')
realGraph = reader.graph
with open('node_colors.txt','ab') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
    writer.writerow(['nodeID', 'layerID', 'color', 'size'])
    for layer_id in xrange(1, 6):
        for node in sorted(realGraph.nodes(), key=lambda n : n.id):
            writer.writerow([node.id + 1, layer_id, 'red' if node.label == 0 else 'blue', 10])