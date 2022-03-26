from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import torch
from sklearn.manifold import LocallyLinearEmbedding

def hdfs_generate(name='myhdfs/hdfs_train', data_dir='../data', window_size=10, bidirectional=False):
    num_sessions = 0
    inputs = []
    outputs = []
    sub_sessions=0
    with open(os.path.join(data_dir, name), 'r') as f:
        for line in f.readlines():
            sub_sessions +=len(line)-window_size+1
            line = list(map(lambda n: n+2, map(int, line.strip().split())))
            line.insert(0, 1)
            if len(line) < window_size // 2:
                continue
            line = line + [0] * (window_size - len(line)-1)
            line.append(2)
            #line = tuple(line)

            for i in range(len(line) - window_size+1):
                inputs.append(line[i:i + window_size])

                if bidirectional:
                    outputs.append(line[i:i + window_size][::-1])
                else:
                    outputs.append(line[i:i + window_size][::-1])

            num_sessions += 1
    print(num_sessions)
    print("Sessions", len(inputs))
    print(sub_sessions)
    return list(inputs),list(outputs)

def hdfs_generate_ori(name='hdfs-new/hdfs_train_all', data_dir='unswnbData', window_size=10, bidirectional=False):
    num_sessions = 0
    inputs = []
    outputs = []
    with open(os.path.join(data_dir, name), 'r') as f:
        for line in f.readlines():
            #if len(line.strip().split())<window_size:
            #    continue
            line = tuple(map(lambda n: n-1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                if bidirectional:
                    outputs.append(line[i:i + window_size][::-1])
                else:
                    outputs.append(line[i:i + window_size][::-1])
                if len(inputs[-1]) < window_size:
                    continue
            num_sessions += 1
    print(num_sessions)
    print("Sessions", len(inputs))
    #inputs = np.expand_dims(inputs, -1)
    #index = np.random.choice(inputs.shape[0], 30000, replace=False)
    #index1 = np.random.choice(inputs.shape[0], 1000, replace=False)
    #totalindex = np.arange(len(inputs))
    #remain_index = np.setdiff1d(totalindex, index)

    return inputs,outputs

def test_normal_generate(name='hdfs-old/hdfs_test_normal', data_dir='unswnbData', window_size=10):
    num_sessions = 0

    inputs = set()
    outputs = set()
    labels=[]

    with open(os.path.join(data_dir, name), 'r') as f:
        for line in f.readlines():

            #if len(line.strip().split()) < window_size:
            #    continue

            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [29] * (window_size - len(line))
            line = tuple(line)


            for i in range(len(line) - window_size):
                inputs.add(tuple(line[i:i + window_size]))
                outputs.add(tuple(line[i:i + window_size][::-1]))

            num_sessions += 1
    #print(num_sessions)
    print("Sessions", len(inputs))
    #inputs = np.expand_dims(inputs, -1)
    #index = np.random.choice(inputs.shape[0], 30000, replace=False)
    #index1 = np.random.choice(inputs.shape[0], 1000, replace=False)
    #totalindex = np.arange(len(inputs))
    #remain_index = np.setdiff1d(totalindex, index)

    return list(inputs), list(outputs)

def test_abnormal_generate(name='hdfs-old/hdfs_test_abnormal', data_dir='unswnbData', window_size=10):
    num_sessions = 0

    inputs = set()
    outputs = set()
    labels=[]

    with open(os.path.join(data_dir, name), 'r') as f:
        for line in f.readlines():

            #if len(line.strip().split()) < window_size:
            #    continue

            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [29] * (window_size - len(line))
            line = tuple(line)

            for i in range(len(line) - window_size):
                inputs.add(tuple(line[i:i + window_size]))
                outputs.add(tuple(line[i:i + window_size][::-1]))

            num_sessions += 1
    #print(num_sessions)
    print("Sessions", len(inputs))
    #inputs = np.expand_dims(inputs, -1)
    #index = np.random.choice(inputs.shape[0], 30000, replace=False)
    #index1 = np.random.choice(inputs.shape[0], 1000, replace=False)
    #totalindex = np.arange(len(inputs))
    #remain_index = np.setdiff1d(totalindex, index)

    return list(inputs), list(outputs)

def test_generate(name, window_size=10):
    hdfs = {}
    # hdfs-old = []
    num_sessions = 0
    sub_sessions=0
    with open(name, 'r') as f:
        for ln in f.readlines():
            #print(ln)
            ln = list(map(lambda n: n+2, map(int, ln.strip().split())))
            ln.insert(0, 1)
            if len(ln)<window_size//2:
                continue
            ln = ln + [0] * (window_size - len(ln)-1)
            ln.append(2)
            #ln = ln + [0] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)]=hdfs.get(tuple(ln),0)+1
            num_sessions +=1
            sub_sessions += len(ln) - window_size + 1
            # hdfs-old.append(tuple(ln))
            #if num_sessions==500:
            #    break
    print(f"Number of sessions({name}): {len(hdfs)}")
    print(num_sessions)
    print(sub_sessions)
    return hdfs,num_sessions

def test_generateforCluster(name, window_size=10):
    inputs = []
    outputs = []
    # hdfs-old = []
    num_sessions = 0
    sub_sessions=0
    num=0
    with open(name, 'r') as f:
        for ln in f.readlines():
            num +=1
            if num>200000:
                break
            sub_sessions +=len(ln)-window_size +1
            #print(ln)
            ln = list(map(lambda n: n+2, map(int, ln.strip().split())))
            ln.insert(0, 1)
            if len(ln)<window_size//2:
                continue
            ln = ln + [0] * (window_size - len(ln)-1)
            ln.append(2)
            for i in range(len(ln) - window_size + 1):
                inputs.append(ln[i:i + window_size])
                outputs.append(ln[i:i + window_size][::-1])
            num_sessions += 1
            # hdfs-old.append(tuple(ln))
            #if num_sessions==500:
            #    break
    print(num_sessions)
    print("Sessions", len(inputs))
    return list(inputs), list(outputs)

def LogResults(line, filename):
    f = open(filename, "a")  # 利用追加模式,参数从w替换为a即可
    line=np.array(line)
    for i in range(len(line)):
        line[i]=int(line[i])+1
    newline=[str(i) for i in line]
    output=' '.join(newline)
    f.write("{}\n".format(output))
    f.close()

def LogProbResults(line, filename):
    f = open(filename, "a")  # 利用追加模式,参数从w替换为a即可
    line=line.squeeze(-1)
    line=line.squeeze()
    prob=line.numpy()
    output=[]
    for p in prob:
        t=float('%.6f' % p)
        output.append(t)
    f.write("{}\n".format(output))
    f.close()

def plotProbCluster(normal_prob, abnormal_prob):

    labels = []
    for i in range(len(normal_prob)):
        labels.append('0')
    for i in range(len(abnormal_prob)):
        labels.append('1')

    z_run = np.vstack((normal_prob.squeeze(), abnormal_prob.squeeze()))
    # z_run=prob.squeeze()

    hex_colors = []
    for _ in np.unique(labels):
        hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

    colors = [hex_colors[int(i)] for i in labels]
    # tsne=TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
    # z_run_tsne = TSNE(perplexity=40, min_grad_norm=1E-12, n_iter=300).fit_transform(z_run)
    # z_run_tsne = TSNE(n_components=2, verbose=2, perplexity=80, n_iter=600).fit_transform(z_run)

    embedding = LocallyLinearEmbedding(n_components=2, eigen_solver='dense')
    z_run_tsne = embedding.fit_transform(z_run)
    plt.scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=colors, marker='*', linewidths=0)
    plt.savefig("hdfs-tsne-cluster-embed-newprobcluster-ab_nor.png")
    plt.show()
