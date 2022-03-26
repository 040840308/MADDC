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
import numpy as np
from sklearn.cluster import KMeans
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.discovery.inductive.variants.im_f.algorithm import Parameters
from pm4py.algo.discovery.heuristics.variants import plusplus as heuristic_plus
from pm4py.visualization.dfg import visualizer
from pm4py.objects.dfg.exporter import exporter as dfg_exporter

def EvaluateModelAccuracy(log, net, initial_marking, final_marking):
    prec = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                     variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    print("Model Precision:%.4f" % prec)
    print("Model Fitness:%.4f" % fitness['log_fitness'])

#load cluster model
savedModelName='HDFS_Cluster_kmeans'
if Path(os.path.join('./models',savedModelName)).exists() and True:
    with open(os.path.join('./models',savedModelName), 'rb') as f:
        clusterModel = pickle.load(f)

cluster_lines={}
line_num=0
data=[]
with open("logs/hdfs_train", 'r') as f:
    for line in f:
        line=line.strip('\n')
        data.append(line)
    f.close()
for line in data:
    line_num +=1
    vector=[0]*29
    line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
    for i in line:
        vector[i]+=1
    pred = clusterModel.predict([vector])
    if not pred[0] in cluster_lines:
        cluster_lines[pred[0]]=[]
        cluster_lines[pred[0]].append(vector)
    else:
        cluster_lines[pred[0]].append(vector)
#print(event_onehot)
#print(len(event_onehot))
#print(len(onehot_lines.values()))
savedDFGDir="./ExportedWorkflow_DFG"
i=0# workflow id
evaluate_flag=True

for key in cluster_lines:
    workflowname="WORKFLOW_"+str(i)
    log = EventLog()
    for line in cluster_lines[key]:
        trace = Trace()
        for step in line:
            event = Event({xes_constants.DEFAULT_NAME_KEY: str(step)})
            trace.append(event)
        log.append(trace)
    print("Beginning DFG Mining for Workflow--{}. Please wait...".format(i))
    dfg, sa, ea = pm4py.discover_directly_follows_graph(log)
    print("DFG Creation Successful!")
    parameters = {pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
    gviz = visualizer.apply(dfg, parameters={"start_activities": sa, "end_activities": ea,
                                             "format": "PNG"})
    # visualizer.view(gviz)
    savedPng = os.path.join('./SavedWorkflowPNGS_DFG', workflowname + ".png")

    dfg_exporter.apply(dfg, os.path.join(savedDFGDir, workflowname + ".dfg"),
                       parameters={"start_activities": sa, "end_activities": ea})
    pn_visualizer.save(gviz, savedPng)
    i = i + 1

