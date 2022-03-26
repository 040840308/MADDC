import os
from tqdm import tqdm
import numpy as np
FP=[]
LSTM_FN=[]
DeepLog_FN=[]
EmbedLSTMVAE_TN=[]
train=[]
with open("AnomalyResults/False-negative_EmbedLSTM.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        LSTM_FN.append(line)

with open("AnomalyResults/False-negative_deeplog.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        DeepLog_FN.append(line)

print(len(LSTM_FN))
print(len(DeepLog_FN))
common_FN=[]
for i in LSTM_FN:
    if i in DeepLog_FN:
        common_FN.append(i)
print(common_FN)

with open("AnomalyResults/True-positive_EmbedLSTMVAE.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        EmbedLSTMVAE_TN.append(line)

for j in common_FN:
    if j in EmbedLSTMVAE_TN:
        newline=[str(i) for i in j]
        t=' '.join(newline)
        print(t)
