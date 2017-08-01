'''
Created on 18 mar 2014

@author: Adek
'''
import sys
import time
from multiprocessing import Process

sys.path.append('/home/apopiel/MuNeG')
from experiments.DecisionFusion import DecisionFusion
if __name__ == '__main__':
    start_time = time.time()
    # groups = [100, 500, 1000]
    # groupSize = [2, 3, 4, 5, 6, 7, 8, 9]
    # groupLabel = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    # probInGroup = [5, 6, 7, 8, 9]
    # probBetweenGroups = [0.1, 0.5, 1, 2, 3, 4, 5]

    # for nodes in groups:
    #     for size in groupSize:
    #         for label in groupLabel:
    #             for probIn in probInGroup:
    #                 for probBetween in probBetweenGroups:
    nodes = int(sys.argv[1])
    size = int(sys.argv[2])
    for label in [5.5]:
        for probIn in [5]:
            for probBetween in [1]:
                for nrOfLayers in [2]:
                    processes = []
                    for fold in [2]:
                        df = DecisionFusion(nodes, size, label, probIn, probBetween, nrOfLayers, fold)
                        p = Process(target=df.processExperiment)
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
    print("--- %s seconds ---" % str(time.time() - start_time))