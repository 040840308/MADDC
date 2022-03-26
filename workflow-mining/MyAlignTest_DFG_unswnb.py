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
from pathlib import Path
import pickle
from hdbscan import approximate_predict

def resolveAlignTraces(alignTracesArray):

    Diagnosis_Map = []

    for trace in alignTracesArray:
        trace_dict={}
        trace_dict['fitness']=trace['cost']
        #print("Fitness Score: %.3f" % trace['fitness'])
        # check missing and redudant events
        # recursively traverse all align tuples in alignTraces, note that does not consider ('>>', None)
        # (event,'>>')represent redudant events
        # ('>>',event)represent missing events
        diagnosis_align=[]
        #(a, b) is align couple
        for (a,b) in trace['alignment']:
            if a=='>>' and b==None:
                continue
            elif a=='>>' and b!='>>':
                diagnosis_align.append(str(b)+'*') # *represent missing event
            elif a!='>>' and b=='>>':
                diagnosis_align.append(str(a)+'#') # #represent redundant event
            elif a==b:
                diagnosis_align.append(str(a))
            elif a!=b:
                print("ERROR!")
                continue
        trace_dict['align'] = str(diagnosis_align)
        Diagnosis_Map.append(trace_dict)

    return Diagnosis_Map

def DFSImporter():
    id = 0
    dfgs = {}
    sas = {}
    eas = {}

    for f in os.listdir("./DFGModels/unswnb/SavedUnswnbWorkflowModels_DFG"):
        dfg, sa, ea = dfg_importer.apply(
            os.path.join("./DFGModels/unswnb/SavedUnswnbWorkflowModels_DFG", f))
        dfgs[id] = dfg
        sas[id] = sa
        eas[id] = ea
        id = id + 1
    return dfgs,sas, eas

def ExtractCluster(line):
    vector=[0] *291
    for i in line:
        vector[i] +=1
    savedModelName='Unswnb_Cluster_hdbscan_5.pkl'
    if Path(os.path.join('./DFGModels/unswnb',savedModelName)).exists():
        with open(os.path.join('./DFGModels/unswnb',savedModelName),'rb') as f:
            clusterModel=pickle.load(f)
    pred,prob=approximate_predict(clusterModel, [vector])
    return pred

def Diagnose_TP_Cluster():
    with open("madd-unswnb-tp-case.txt", 'r') as f:
        for line in f.readlines():
            print(line)
            Total_Align_Diagnosis_Map = []
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            pred=ExtractCluster(line)
            dfg_name="WORKFLOW_"+str(pred[0])+".dfg"
            dfg, sa, ea = dfg_importer.apply(
                os.path.join("./DFGModels/unswnb/SavedUnswnbWorkflowModels_DFG", dfg_name))
            log = EventLog()
            trace = Trace()
            for e in line:
                event = Event({xes_constants.DEFAULT_NAME_KEY: str(e)})
                trace.append(event)
            log.append(trace)
            try:
                aligned_traces = dfg_alignment.apply(log, dfg, sa, ea)
                #print(aligned_traces)
                diagnosis_aligntraces_Map_array = resolveAlignTraces(aligned_traces)
                for d in diagnosis_aligntraces_Map_array:
                    if not d in Total_Align_Diagnosis_Map:
                        Total_Align_Diagnosis_Map.append(d)
            except BaseException:
                continue
            #print("Origianl Sequence: %s " % str(line))
            print("Origianl Sequence: %s " % str(line))
            print("Diagnosis As Follows!")
            sorted_diagnosis_align_map = sorted(Total_Align_Diagnosis_Map, key=lambda k: k['fitness'], reverse=False)
            for alin in sorted_diagnosis_align_map:
                print(alin)
            fitness_array = [alin['fitness'] for alin in sorted_diagnosis_align_map]
            print(fitness_array[0])

def Diagnose_TP():#generate diagnosis result for each anomaly
    dfgs, sas, eas = DFSImporter()
    with open("madd-unswnb-tp-case.txt", 'r') as f:
        for line in f.readlines():
            print(line)
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            # line=list(line).remove(1)
            # line=list(line).remove(2)
            # line = list(map(lambda n: n - 2, line))
            log = EventLog()
            trace = Trace()
            for e in line:
                event = Event({xes_constants.DEFAULT_NAME_KEY: str(e)})
                trace.append(event)
            log.append(trace)
            Total_Align_Diagnosis_Map = []
            # start to perform comforance checking
            # fp_net = footprints_discovery.apply(net, im, fm)
            # The footprints of the Petri net model can be calculated as follows
            for k in dfgs.keys():
                dfg = dfgs[k]
                sa = sas[k]
                ea = eas[k]
                try:
                    aligned_traces = dfg_alignment.apply(log, dfg, sa, ea)
                    # print(aligned_traces)
                    diagnosis_aligntraces_Map_array = resolveAlignTraces(aligned_traces)
                    for d in diagnosis_aligntraces_Map_array:
                        if not d in Total_Align_Diagnosis_Map:
                            Total_Align_Diagnosis_Map.append(d)
                except BaseException:
                    continue
            print("Origianl Sequence: %s " % str(line))
            print("Diagnosis As Follows!")
            sorted_diagnosis_align_map = sorted(Total_Align_Diagnosis_Map, key=lambda k: k['fitness'], reverse=False)
            for alin in sorted_diagnosis_align_map:
                print(alin)
            fitness_array = [alin['fitness'] for alin in sorted_diagnosis_align_map]
            print(fitness_array[0])

if __name__ == "__main__":
    #Diagnose_FP()
    #Diagnose_TP()
    Diagnose_TP_Cluster()


