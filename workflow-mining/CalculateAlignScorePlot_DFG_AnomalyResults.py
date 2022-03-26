import os
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.util import xes_constants
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.dfg.importer import importer as dfg_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.discovery.footprints import algorithm as fp_discovery
from pm4py.algo.conformance import alignments as ali
from pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star import Parameters
from pm4py.objects import log as log_lib
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.visualization.align_table import visualizer
# import all perinet pnl file
from pm4py.algo.conformance.alignments.decomposed import algorithm as decomp_alignments
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from random import randint
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def resolveAlignTraces(alignTracesArray):
    min_score=10000000000

    for trace in alignTracesArray:
        if trace['cost']<min_score:
            min_score=trace['cost']

    return min_score

id=0
dfgs={}
sas={}
eas={}

for f in os.listdir("./ExportedWorkflow_DFG"):
    dfg, sa, ea = dfg_importer.apply(
        os.path.join("./ExportedWorkflow_DFG", f))
    dfgs[id]=dfg
    sas[id]=sa
    eas[id]=ea
    id=id+1

pred=[]
#extract normal scores
normal_fitness_score=[]
total_fitness_score=[]
print("calculate normal align score!")
with open("AnomalyResults/False-Positive_deeplog.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
        log = EventLog()
        trace = Trace()
        for e in line:
            event = Event({xes_constants.DEFAULT_NAME_KEY: str(e)})
            trace.append(event)
        log.append(trace)
        #line_score = 0
        line_score = 0.00000000000
        for k in dfgs.keys():
            dfg = dfgs[k]
            sa = sas[k]
            ea = eas[k]
            try:
                aligned_traces = dfg_alignment.apply(log, dfg, sa, ea)
                fitness_score = resolveAlignTraces(aligned_traces)
                if fitness_score > line_score:
                    line_score = fitness_score
            except BaseException:
                continue
        total_fitness_score.append(line_score)
        pred.append(0)

#extract abnormal scores
abnormal_fitness_score=[]
print("calculate abnormal align score!")
with open("AnomalyResults/False-negative_deeplog.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
        log = EventLog()
        trace = Trace()
        for e in line:
            event = Event({xes_constants.DEFAULT_NAME_KEY: str(e)})
            trace.append(event)
        log.append(trace)
        line_score = 0.00000000000
        for k in dfgs.keys():
            dfg = dfgs[k]
            sa = sas[k]
            ea = eas[k]
            try:
                aligned_traces = dfg_alignment.apply(log, dfg, sa, ea)
                fitness_score = resolveAlignTraces(aligned_traces)
                if fitness_score > line_score:
                    line_score = fitness_score
            except BaseException:
                continue
        total_fitness_score.append(line_score)
        pred.append(1)

#max_score=0
#if max(normal_fitness_score)>max_score:
#    max_score=max(normal_fitness_score)
#if max(abnormal_fitness_score)>max_score:
#    max_score=max(abnormal_fitness_score)

#new_normal_fitness_score=[float(i)/max_score for i in normal_fitness_score]
#new_abnormal_fitness_score=[float(i)/max_score for i in abnormal_fitness_score]
print("Max fitness score: %d, Min fitness score:%d " %(max(total_fitness_score), min(total_fitness_score)))
new_total_fitness_score=[float(i)/max(total_fitness_score) for i in total_fitness_score]
#print("Max fitness score: %.4f, Min fitness score:%.4f " %(max(new_total_fitness_score), min(new_total_fitness_score)))
#plot cluster
hex_colors = []
for _ in np.unique(pred):
    hex_colors.append('#%06X' % randint(0, 0xFFFFFF))
colors = [hex_colors[int(i)] for i in pred]
#print(new_total_fitness_score)
f = open("fitness_deeplog_dfg.txt", "a")  # 利用追加模式,参数从w替换为a即可
newline = [str(i) for i in new_total_fitness_score]
output = ' '.join(newline)
f.write("{}\n".format(output))
f.close()
#z_run_tsne = TruncatedSVD(n_components=3).fit_transform(event_vector)
plt.scatter(np.arange(0,len(new_total_fitness_score),1), new_total_fitness_score, c=colors, marker='*', linewidths=0)
plt.savefig("alignscore_dfg_deeplog.png")
plt.show()
