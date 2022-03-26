import os
from tqdm import tqdm
FP=[]
FN=[]
train=[]

with open("AnomalyResults/common_abnormal_normal", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        train.append(line)

with open("AnomalyResults/False-Positive_deeplog.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        FP.append(line)

with open("AnomalyResults/False-negative_deeplog.txt", 'r') as f:
    for line in tqdm(f.readlines()):
        line = tuple(map(lambda n: n, map(int, line.strip().split())))
        FN.append(line)

print(len(FN))
matched_fn=[]
for fn in FN:
    if fn in train:
        matched_fn.append(fn)
print(len(matched_fn))


print(len(FP))
matched_fp=[]
for fp in FP:
    if fp in train:
        matched_fp.append(fp)
print(len(matched_fp))



