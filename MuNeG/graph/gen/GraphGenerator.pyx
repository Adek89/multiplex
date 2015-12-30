'''
Created on 6 lut 2014

@author: Adek
'''

import random as rand

import networkx as nx
cimport cython
cimport graph.gen.Group as g
cimport graph.gen.Node as node
cdef class GraphGenerator:
    '''
    classdocs
    '''
    cdef int n0
    cdef int gs
    cdef int ng
    cdef list groups
    cdef graph
    cdef list nodes
    
    cdef list reds
    cdef list blues

    cdef list layerWeights
    cdef int probEdgeInSameGroup
    cdef int probEdgeBetweenOtherGroups
    cdef int grLabelHomogenity
    cdef list layerName

    def __cinit__(self, int n0, int gs, list layerWeights, int grLabelHomogenity, int probEdgeInSameGroup, int probEdgeBetweenOtherGroups, list layerName):
        self.n0 = n0
        self.gs = gs
        self.ng = n0/gs
        self.layerWeights = layerWeights
        self.grLabelHomogenity = grLabelHomogenity
        self.probEdgeInSameGroup = probEdgeInSameGroup
        self.probEdgeBetweenOtherGroups = probEdgeBetweenOtherGroups
        self.layerName = layerName
        self.nodes = []
        self.graph = nx.MultiGraph()
        self.groups = []
        self.reds = []
        self.blues = []

    cdef prepareGroups(self):
        cdef int i
        cdef gType
        for i in range(0, self.ng):
            gType = rand.randint(0,1)
            if (gType == 0):
                self.groups.append(g.Group('red', i))
            else: 
                self.groups.append(g.Group('blue', i))
            
    cpdef generate(self):
        self.prepareGroups()
        self.generateNodes()
        self.generateEdges()
     #   print(self.graph[self.nodes.__getitem__(22)])
        
#         position = nx.spectral_layout(self.graph)
        '''
        position = nx.circular_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, position, self.reds, node_color = 'r')
        nx.draw_networkx_nodes(self.graph, position, self.blues, node_color = 'b')
        nx.draw_networkx_edges(self.graph, position)
        nx.draw_networkx_labels(self.graph, position)
        plt.show()
#        nx.write_pajek(self.graph, 'E:\Eclipse_workspace\Generator_py\output.temp')
        '''
#         directory = dirname(__file__)
#         nx.write_adjlist(self.graph, directory + "\\adj_list" + str(i) + ".txt")
        return self.graph
        
        
        
    cdef generateNodes(self):
        cdef int j = 0
        cdef int label = 0
        cdef int i
        cdef group
        cdef gType
        for i in range(0, self.n0):
            group = self.groups.__getitem__(j)
            gType = group.gType
            if (gType == 'red'):
                self.addNodeToCorrectCollection(self.grLabelHomogenity, 10-self.grLabelHomogenity, group, i, self.reds)
            else: 
                self.addNodeToCorrectCollection(10-self.grLabelHomogenity, self.grLabelHomogenity, group, i, self.blues)
            j = j + 1
            if (j > 9 or j > self.ng-1):
                j = 0
                
    cdef void addNodeToCorrectCollection(self, int lowerWeight, int higherWeight, g.Group group, int i, list nodeCollection):
        cdef list weights = [lowerWeight, higherWeight]
        cdef int label = self.weighted_choice(weights)
        n = node.Node(group, label, i)
        nodeCollection.append(n)     
        self.nodes.append(n)
        self.graph.add_node(n)
                
    cdef generateEdges(self ):
        cdef int w
        cdef float probInSame
        cdef float probOthers
        cdef int currentLayerWeight
        cdef str currentLayerName
        cdef int i
        cdef int j
        cdef node1
        cdef node2
        cdef list weights
        cdef int isEdgeExist
        for w in range (0, self.layerWeights.__len__()):
            probInSame = self.probEdgeInSameGroup
            probOthers = self.probEdgeBetweenOtherGroups
            currentLayerWeight = self.layerWeights.__getitem__(w)
            currentLayerName = self.layerName.__getitem__(w)
            for i in range(0, self.n0): 
                node1 = self.nodes.__getitem__(i) 
                for j in range(i+1, self.n0):
                    node2 = self.nodes.__getitem__(j)
                    if (node1.group == node2.group):
                        weights = [10-probInSame, probInSame]
                        isEdgeExist = self.weighted_choice(weights)
                    else:
                        weights = [10-probOthers, probOthers]
                        isEdgeExist = self.weighted_choice(weights)
                    print "edge exists: " + str(isEdgeExist)
                    if (isEdgeExist == 1):
                        self.graph.add_edge(node1, node2, weight=currentLayerWeight, layer=currentLayerName, conWeight = 1)  #conWeight - waga polaczenia
                        
    cdef int weighted_choice(self, list weights):
        cdef list totals = []
        cdef int running_total = 0
        cdef int w
        cdef float rnd
        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = rand.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i           
            
    