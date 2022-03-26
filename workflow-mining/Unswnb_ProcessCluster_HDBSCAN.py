import os
import pm4py
import hdbscan
from hdbscan import HDBSCAN, approximate_predict
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import xes_constants
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from sklearn.datasets import load_iris
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from random import randint
from sklearn.manifold import TSNE
from pathlib import Path

event_vector=[]
vector_lines={}
savedModelName='Unswnb_Cluster_hdbscan.pkl'

with open("D:/UNSWNB15-Data/Sequences/Train/59.166.0.0/sequences.txt", 'r') as f:
    for line in f.readlines():
        vector = [0] * 706
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        for i in line:
            vector[i] += 1
        event_vector.append(vector)

if Path(os.path.join('./models',savedModelName)).exists() and False:
    with open(os.path.join('./models',savedModelName), 'rb') as f:
        clusterModel = pickle.load(f)
        labels, probas = approximate_predict(clusterModel, event_vector)
else:
    clusterModel = HDBSCAN(min_cluster_size=5, prediction_data=True).fit(event_vector)
    # Fitting
    #cluster_labels = clusterModel.fit(event_vector)
    #clusterModel.a
    # 存储模型
    with open(os.path.join('./models',savedModelName), 'wb') as f:
        pickle.dump(clusterModel, f)
    labels, probas = approximate_predict(clusterModel, event_vector)
    #print(labels)
    #print(probas)


#plot cluster
hex_colors = []
for _ in np.unique(labels):
    hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
colors = [hex_colors[int(i)] for i in labels]
#z_run_tsne = TruncatedSVD(n_components=3).fit_transform(event_vector)
z_run_tsne = TSNE(n_components=2,perplexity=50,n_iter=2000).fit_transform(event_vector)
plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
plt.savefig("cluster_hdbscan_unswnb.png")
plt.show()
