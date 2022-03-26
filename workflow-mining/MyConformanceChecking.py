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
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.discovery.footprints import algorithm as fp_discovery

# import all perinet pnl file

id=0
nets={}
initialMarking={}
finalMarking={}

for f in os.listdir("./ExportedWorkflow_Perinet"):
    net, initial_marking, final_marking = pnml_importer.apply(
        os.path.join("./ExportedWorkflow_Perinet", f))
    nets[id]=net
    initialMarking[id]=initial_marking
    finalMarking[id]=final_marking
    id=id+1

window_size=10

with open("logs/hdfs_test.txt", 'r') as f:
    for line in f.readlines():
        line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
        log = EventLog()
        for i in range(len(line) - window_size):
            trace = Trace()
            for e in line[i:i + window_size]:
                event = Event({xes_constants.DEFAULT_NAME_KEY: str(e)})
                trace.append(event)
            log.append(trace)

            #start to perform comforance checking
            #fp_net = footprints_discovery.apply(net, im, fm)
            # The footprints of the Petri net model can be calculated as follows
            for k in nets.keys():
                net=nets[k]
                initial_marking=initialMarking[k]
                final_marking=finalMarking[k]
                fp_net = fp_discovery.apply(net, initial_marking, final_marking)
                '''try:
                    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)
                except BaseException:
                    continue
                log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)
                print(log_fitness)'''
                try:
                    token_replay_fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
                                                         variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
                except BaseException:
                    continue
                print("Token_replay_Fitness: %.3f" % token_replay_fitness['log_fitness'])
                try:
                    alignment_fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking,
                                                         variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
                except BaseException:
                    continue
                #print(alignment_fitness)
                print("Alignment_Fitness: %.3f" % alignment_fitness['averageFitness'])
