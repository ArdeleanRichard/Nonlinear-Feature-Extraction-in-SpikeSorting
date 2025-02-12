import os

import numpy as np
from validation.rank_aggregation.rank_agg2.rankagg import FullListRankAggregator

os.chdir("../")

METRICS = ["ARI", "AMI", "VM", "DBS", "CHS", "SS"]
METHODS = ["PCA", "ICA", "Isomap", "AE"]



pca =               np.loadtxt(f"./results/pca.csv", dtype=float, delimiter=",")
ica =               np.loadtxt(f"./results/ica.csv", dtype=float, delimiter=",")
isomap =            np.loadtxt(f"./results/isomap.csv", dtype=float, delimiter=",")
ae_normal =         np.loadtxt(f"./results/ae_normal.csv", dtype=float, delimiter=",")




for met_id, met in enumerate(METRICS):
    print("----------------------------------------------------------------")
    print(f"------------------------------{met}-----------------------------")
    print("----------------------------------------------------------------")

    ranks_list = []
    ranks_dict = []
    for id in range(len(pca)):
        method_results = np.array([pca[id][met_id],
                          ica[id][met_id],
                          isomap[id][met_id],
                          ae_normal[id][met_id],
                          ])
        ranks_list.append(np.array(METHODS)[np.argsort(method_results)[::-1]].tolist())
        # print(ranks_list)

        ranks = {}
        for id, value in enumerate(method_results):
            ranks[METHODS[id]] = value
        ranks_dict.append(ranks)

    FLRA = FullListRankAggregator()
    print("Borda: ", FLRA.aggregate_ranks(ranks_dict, method='borda')[1])
    # print("Spearman: ", FLRA.aggregate_ranks(ranks_dict, method='spearman'))
    print()
    print()
    print()
