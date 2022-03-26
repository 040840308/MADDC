import os
from tqdm import tqdm
import numpy as np
FP=[]
LSTM_FN=[]
LSTMVAE_FN=[]
train=[]

with open("AnomalyResults/hdfs_train", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        train.append(line)

with open("AnomalyResults/False-negative_EmbedLSTM.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        LSTM_FN.append(line)

with open("AnomalyResults/False-negative_EmbedLSTMVAE.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        LSTMVAE_FN.append(line)

print(len(LSTM_FN))
print(len(LSTMVAE_FN))
reduced_FN=[]
for i in LSTM_FN:
    if not i in LSTMVAE_FN:
        reduced_FN.append(i)
added_FN=[]
for i in LSTMVAE_FN:
    if not i in LSTM_FN:
        added_FN.append(i)

common_FN=[]
for i in LSTM_FN:
    if i in LSTMVAE_FN:
        common_FN.append(i)
print(reduced_FN)
print(len(reduced_FN))
print(len(added_FN))
count_FN={}
for i in train:
    if i in reduced_FN:
        if not i in count_FN:
            count_FN[i]=1
        else:
            count_FN[i] +=1
count_FN=sorted(count_FN.items(), key=lambda item: item[1], reverse=True)
print(count_FN)

f = open("AnomalyResults/reducedFN.txt", "a")  # 利用追加模式,参数从w替换为a即可
for fn in reduced_FN:
    line = np.array(fn)
    newline = [str(i) for i in line]
    output = ' '.join(newline)
    f.write("{}\n".format(output))
f.close()