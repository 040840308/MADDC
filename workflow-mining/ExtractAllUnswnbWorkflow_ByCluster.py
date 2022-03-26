import os
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import xes_constants
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pathlib import Path
import pickle
from sklearn.cluster import KMeans
import numpy as np
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator

def EvaluateModelAccuracy(log, net, initial_marking, final_marking):
    prec = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                     variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    print("Model Precision:%.4f" % prec)
    print("Model Fitness:%.4f" % fitness)


class SequenceClustering:
    def __init__(self, method='kmeans'):
        if method not in ['kmeans']:
            raise ValueError(f'Only "kmeans" method is supported, but received: "{method}"')
        self._method = method
        self._model = None

    def InitialCentroid(self,x, K):
        x=np.array(x)
        c0_idx = int(np.random.uniform(0, len(x)))
        centroid = x[c0_idx].reshape(1, -1)  # 选择第一个簇中心
        k = 1
        n = x.shape[0]
        while k < K:
            d2 = []
            for i in range(n):
                subs = centroid - x[i, :]
                dimension2 = np.power(subs, 2)
                dimension_s = np.sum(dimension2, axis=1)  # sum of each row
                d2.append(np.min(dimension_s))
            new_c_idx = np.argmax(d2)
            centroid = np.vstack([centroid, x[new_c_idx]])
            k += 1
        return centroid

    def fit(self, embeddings, min_cluster_num=50, max_cluster_num=4, random_state=9):
        """
        Trains the model and searches for the optimal number of clusters.
        Parameters
        ----------
        embeddings: array-like of number, shape=[number of objects, vector dimension]
            List of vectorized objects.
        min_cluster_num : int, default=2
            Minimum number of clusters.
        max_cluster_num : int, default=4
            Maximum number of clusters.
        random_state: int, default=42
        Returns
        -------
        self
        """
        models = {}

        if self._method == 'kmeans':
            scores = []
            for k in range(min_cluster_num, max_cluster_num + 1):
                centroid=self.InitialCentroid(embeddings,k)
                print("Test cluster %d" % k)
                kmeans = KMeans(n_clusters=k, init='k-means++',random_state=random_state).fit(embeddings)
                scores.append(kmeans.inertia_)
                models[k - min_cluster_num] = kmeans

            selected=np.argmax([scores[i - 1] / scores[i]
                                            for i in range(1, max_cluster_num - min_cluster_num + 1)])

            print("Selected Models %d" % selected)

            self._model = models[selected]

        return self

    def predict(self, embeddings):
        """
        Predict the clusters for given vectorized objects using trained algorithm.
        Parameters
        ----------
        embeddings: array-like of number, shape=[number of objects, vector dimension]
            List of vectorized objects.
        Returns
        -------
        labels: array-like of int, shape=[number of objects]
            Labels of the clusters.
        """
        return self._model.predict(embeddings)

#load cluster model
savedModelName='Unswnb_Cluster.pkl'
if Path(os.path.join('./models',savedModelName)).exists() and True:
    with open(os.path.join('./models',savedModelName), 'rb') as f:
        clusterModel = pickle.load(f)

cluster_lines={}
line_num=0
data=[]
with open("D:/UNSWNB15-Data/Sequences/Train/59.166.0.0/sequences.txt", 'r') as f:
    for line in f:
        line=line.strip('\n')
        data.append(line)
    f.close()
for line in data:
    line_num +=1
    vector=[0]*706
    line = tuple(map(lambda n: n, map(int, line.strip().split())))
    for i in line:
        vector[i] +=1
    pred = clusterModel.predict([vector])

    if not pred[0] in cluster_lines:
        cluster_lines[pred[0]] = []
        cluster_lines[pred[0]].append(vector)
    else:
        cluster_lines[pred[0]].append(vector)
#print(event_onehot)
#print(len(event_onehot))
#print(len(onehot_lines.values()))
#savedPerinetDir="./ExportedWorkflow_Perinet"
i=0# workflow id
print(line_num)
for key in cluster_lines:
    workflowname="WORKFLOW_"+str(i)
    log = EventLog()
    for line in cluster_lines[key]:
        trace = Trace()
        for step in line:
            event = Event({xes_constants.DEFAULT_NAME_KEY: str(step)})
            trace.append(event)
        log.append(trace)
    print("Beginning Petri net Mining for Workflow--{}. Please wait...".format(i))
    net, initial_marking, final_marking = heuristics_miner.apply(log)
    EvaluateModelAccuracy(log, net, initial_marking, final_marking)
    print("Petri Net Creation Successful!")
    parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
    gviz = pn_visualizer.apply(net, initial_marking, final_marking,
                               parameters=parameters,
                               log=log)
    savedPng=os.path.join('./SavedUnswnbWorkflowPNGS',workflowname+".png")

    #pnml_exporter.apply(net, initial_marking, os.path.join(savedPerinetDir,workflowname+".pnml"), final_marking=final_marking)
    pn_visualizer.save(gviz,savedPng)
    i=i+1

