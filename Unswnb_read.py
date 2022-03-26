from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.manifold import LocallyLinearEmbedding

def train_generate(name='Train', data_dir='/home/xiaolei/PycharmWorkspace/MADD/src/data/Sequences/', window_size=10, bidirectional=False):
    #N = 20000
    #if name == 'Test_Normal':
    #    N = 30000
    #if name=='Train':
    #    N=30000
    num_sessions = 0
    inputs = []
    outputs = []
    rootdir=os.path.join(data_dir, name)
    dirs = os.listdir(rootdir)
    for subdir in dirs:
        filename=os.path.join(rootdir, subdir)+"/sequences.txt"
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = list(map(lambda n: n, map(int, line.strip().split())))
                line.insert(0,1)
                if len(line) < window_size//2:
                    continue
                line = line + [0] * (window_size - len(line)-1)
                line.append(2)
                for i in range(len(line) - window_size+1):
                    #if line[i:i + window_size] not in inputs:
                    inputs.append(tuple(line[i:i + window_size]))
                    #if bidirectional:
                    #    outputs.append(line[i:i + window_size][::-1])
                    #else:
                    #    outputs.append(line[i:i + window_size][::-1])
                num_sessions += 1

                #if num_sessions==500:
                #    break
    print(num_sessions)
    print("Sessions", len(inputs))
    print("Sessions", len(list(set(inputs))))
    #import random
    #return random.sample(list(set(inputs)),N)#,random.sample(list(outputs),300000)
    return list(inputs)#, list(outputs)

def test_generate(name='Test_Normal', data_dir='/home/xiaolei/PycharmWorkspace/MADD/src/data/Sequences/', window_size=10):
    data = {}
    num_sessions = 0
    sub_sesssions=0
    rootdir = os.path.join(data_dir, name)
    dirs = os.listdir(rootdir)
    for subdir in dirs:
        filename = os.path.join(rootdir, subdir) + "/sequences.txt"
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = list(map(lambda n: n, map(int, line.strip().split())))
                line.insert(0, 1)
                if len(line)<window_size//2:
                    continue
                line = line + [0] * (window_size - len(line)-1)
                line.append(2)
                sub_sesssions +=len(line)-window_size+1
                data[tuple(line)]=data.get(tuple(line),0)+1
                num_sessions +=1
    print(f"Number of sessions({name}): {len(data)}")
    print(num_sessions)
    print(sub_sesssions)
    return data

def test_generate_forcluster(name='Test_Normal', data_dir='/home/xiaolei/PycharmWorkspace/MADD/src/data/Sequences/', window_size=10):
    num_sessions = 0
    inputs = []
    outputs = []
    rootdir = os.path.join(data_dir, name)
    dirs = os.listdir(rootdir)
    for subdir in dirs:
        filename = os.path.join(rootdir, subdir) + "/sequences.txt"
        # if num_sessions == 500:
        #    continue
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = list(map(lambda n: n, map(int, line.strip().split())))
                line.insert(0, 1)
                if len(line) < window_size // 2:
                    continue
                line = line + [0] * (window_size - len(line) - 1)
                line.append(2)
                for i in range(len(line) - window_size + 1):
                    inputs.append(line[i:i + window_size])
                    outputs.append(line[i:i + window_size][::-1])
                num_sessions += 1

                # if num_sessions==500:
                #    break
    print(num_sessions)
    print("Sessions", len(inputs))
    return list(inputs), list(outputs)