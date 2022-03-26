from EmbedLSTMVAE import EmbedLSTMVAE
import utils
import numpy as np
from numpy import *
import torch
torch.cuda.empty_cache()
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import time
import Unswnb_read
import logging
from random import randint
from tqdm import tqdm
import pickle

dload='./model_dir'
options={}
#hyper parameters
hidden_size = 128
options['hidden_size']=hidden_size
hidden_layer_depth = 2
options['hidden_layer_depth']=hidden_layer_depth
latent_length = 30
options['latent_length']=latent_length
batch_size = 1024
options['batch_size']=batch_size
learning_rate = 0.006 #modify default to 0.005,0.005, rec 2, kl 8
options['learning_rate']=learning_rate
lr_step = (180,230)
options['lr_step']=lr_step
lr_decay_ratio = 0.1
options['lr_decay_ration']=lr_decay_ratio
n_epochs = 100
options['n_epochs']=n_epochs
num_keys=291
options['num_keys']=num_keys
embed_dim=512
options['embed_dim']=embed_dim
dropout_rate = 0.2
options['dropout_rate']=dropout_rate
optimizer = 'Adam' # options: ADAM, SGD
options['optimizer']=optimizer
cuda = True # options: True, False
options['cuda']=cuda
print_every=100
options['print_every']=print_every
clip = True # options: True, False
options['clip']=clip
max_grad_norm=5
options['max_grad_norm']=max_grad_norm
loss = 'CrossEntropyLoss' # options: SmoothL1Loss, MSELoss
options['loss']=loss
block = 'LSTM' # options: LSTM, GRU
options['block']=block
options['save_dir']="./result/unswnb/"

#X_train, X_test = utils.hdfs_generate()
#X_train=inputs[:10000]
#train_dataset = TensorDataset(torch.from_numpy(X_train))
#test_dataset = TensorDataset(torch.from_numpy(X_test))
sequence_length = 10
number_of_features = 1
from numpy import percentile
bidirectional=False
#TopN=int(num_keys*1/29)
AnomalyDetect_TopN=True
import gc

#Anomaly_Threshold=0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_normal_loader = Unswnb_read.test_generate(name='Test_Normal')
test_abnormal_loader = Unswnb_read.test_generate(name='Test_Abnormal')

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def TransformLineToSeqs(line,window_size=10):
    inputs = []
    line=list(line)
    line = line + [707] * (window_size - len(line))
    for i in range(len(line) - window_size):
        inputs.append(line[i:i + window_size])
    return list(inputs)

def NewEvaluateAccuracy_EventLevel_Batch(savedmodel,TopN):
    resultdir = "./UNSWNB_Results/TOPN-" + str(TopN)
    os.makedirs(resultdir, exist_ok=True)
    fp_file = resultdir + "/False-Positive_unswnb_MADD.txt"
    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
    f.close()
    fn_file = resultdir + "/False-negative_unswnb_MADD.txt"
    f = open(fn_file, "a")
    f.close()
    final_result = resultdir + "/statistics_unswnb_MADD.txt"
    f = open(final_result, "a")
    f.close()
    savedmodel.eval()
    FP1 = 0
    FP2 = 0
    TP1 = 0
    TP2 = 0
    window_size = 10
    num_session = 0
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            line_num=test_normal_loader[line]
            num_session += line_num
            seqs=TransformLineToSeqs(line)
            input_x = torch.tensor(seqs, dtype=torch.int).view(
                -1, window_size).to(device)
            #input_x = input_x.squeeze(-1)
            x_decoded, latent= savedmodel.forward(input_x)
            gc.collect()
            rec_x = x_decoded
            predicted = torch.argsort(rec_x, -1)[:, :, -TopN:] #(10,218,73)
            #predicted=predicted.permute(1,0,2)
            sh=list(predicted.size())
            flag = False
            if sh[1]==len(seqs):
                for i in range(len(seqs)):
                    tmp_predicted=predicted[:,i,:]
                    tmp_input_x=input_x[i,:]
                    tmp_final_predict = tmp_predicted.squeeze(-2).int().tolist()[::-1]
                    for j in range(sh[0]):
                        if not tmp_input_x[j] in tmp_final_predict[j]:
                            flag = True
                            print("False Positive!")
                            FP1 += 1
                            FP2 += line_num
                            f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
                            line = np.array(line)
                            for i in range(len(line)):
                                line[i] = int(line[i]) + 1
                            newline = [str(i) for i in line]
                            output = ' '.join(newline)
                            f.write("{}\n".format(output))
                            f.close()
                            break
                    if flag:
                        break

    TN = num_session - FP2
    #logging.info(f'False positive (FP): {FP1}')
    #print(f'False positive (FP): {FP1}')
    #logging.info(f'False positive (FP): {FP2}')
    #print(f'False positive (FP): {FP2}')
    #logging.info(f'True negative (TN): {TN}')
    #print(f'True negative (TN): {TN}')

    num_session = 0
    with torch.no_grad():
        for line in tqdm(test_abnormal_loader.keys()):
            line_num=test_abnormal_loader[line]
            num_session += line_num
            seqs = TransformLineToSeqs(line)
            input_x = torch.tensor(seqs, dtype=torch.int).view(
                -1, window_size).to(device)
            input_x = input_x.squeeze(-1)
            x_decoded, latent= savedmodel.forward(input_x)
            gc.collect()
            rec_x = x_decoded
            predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]  # (10,218,73)
            sh = list(predicted.size())
            flag = False
            if sh[1]==len(seqs):
                for i in range(len(seqs)):
                    tmp_predicted=predicted[:,i,:]
                    tmp_input_x=input_x[i,:]
                    tmp_final_predict = tmp_predicted.squeeze(-2).int().tolist()[::-1]
                    for j in range(sh[0]):
                        if not tmp_input_x[j] in tmp_final_predict[j]:
                            flag = True
                            TP1 += 1
                            TP2 += line_num
                            break
                    if flag:
                        break
            if not flag:
                f = open(fn_file, "a")  # 利用追加模式,参数从w替换为a即可
                line = np.array(line)
                for i in range(len(line)):
                    line[i] = int(line[i]) + 1
                newline = [str(i) for i in line]
                output = ' '.join(newline)
                f.write("{}\n".format(output))
                f.close()
    FN = num_session - TP2
    #logging.info(f'True positive (TP): {TP2}')
    #print(f'True positive (TP): {TP2}')
    #logging.info(f'True positive (TP): {TP1}')
    #print(f'True positive (TP): {TP1}')
    #logging.info(f'False negative (FN): {FN}')
    #print(f'False negative (FN): {FN}')
    f = open(final_result, "a")  # 利用追加模式,参数从w替换为a即可
    str1 = f'False positive (FP2): {FP2}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP2}'
    f.write(str1 + "\r\n")
    P = 100 * TP2 / (TP2 + FP2)
    R = 100 * TP2 / (TP2 + FN)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP2, FN, P, R, F1)
    f.write(str1 + "\r\n")
    f.close()
    print('Finished Predicting')

def ExtractProb(x_decoded, Y):
    # extract probability
    softmax_out = F.softmax(x_decoded, dim=-1)
    y = Y.tolist()
    softmax_out = softmax_out.tolist()
    df_softmax = pd.DataFrame(softmax_out)
    df_y = pd.DataFrame(y)
    df = df_softmax + df_y
    df = df.applymap(lambda x: x[int(x[num_keys])])  # num_keys is to extract the index probability, 1+num_keys-1
    prob = np.array(df)
    reconstruct_probability = torch.tensor(prob)
    reconstruct_probability = reconstruct_probability.unsqueeze(-1)

    return reconstruct_probability

def EvaluateAccuracy_EventLevel(savedmodel,TopN, k):
    Anomaly_Threshold=k
    resultdir = "./UNSWNB_Results/TOPN-" + str(TopN)+"_"+str(Anomaly_Threshold)
    os.makedirs(resultdir, exist_ok=True)
    fp_file = resultdir + "/False-Positive_unswnb_MADD.txt"
    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
    f.close()
    fn_file = resultdir + "/False-negative_unswnb_MADD.txt"
    f = open(fn_file, "a")
    f.close()
    final_result = resultdir + "/statistics_unswnb_MADD.txt"
    f = open(final_result, "a")
    f.close()
    # load the model from disk
    #cluster_model = pickle.load(open("./result/unswnb/MADD_LOCALCLUSTER.pkl", 'rb'))
    savedmodel.eval()
    FP1 = 0
    FP2 = 0
    TP1 = 0
    TP2 = 0
    window_size = 10
    num_session = 0
    total_probs_abnormal = []

    with torch.no_grad():
        for line in tqdm(test_abnormal_loader.keys()):
            # print(len(line))
            num_session += test_abnormal_loader[line]
            linenumber = test_abnormal_loader[line]
            flag = False
            for i in range(len(line) - window_size + 1):
                if flag:
                    break
                seq = line[i:i + window_size]
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob = ExtractProb(x_decoded, input_y)

                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    # print(final_probs)
                    for k in range(len(final_predict)):
                        # print(float(final_probs[k][0]))
                        if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        #if float(final_probs[k][0]) < Anomaly_Threshold:
                            TP1 += 1
                            TP2 += linenumber
                            flag = True
                            break
                        #else:
                        #    print(float(final_probs[k][0]))
            if not flag:
                print("False Negative!")
                f = open(fn_file, "a")  # 利用追加模式,参数从w替换为a即可
                line = np.array(line)
                for i in range(len(line)):
                    line[i] = int(line[i])
                newline = [str(i) for i in line]
                output = ' '.join(newline)
                f.write("{}\n".format(output))
                f.close()
    FN = num_session - TP2
    logging.info(f'False negative (FN): {FN}')
    print(f'False negative (FN): {FN}')

    num_session = 0
    total_outlier_score=[]
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            line_num = test_normal_loader[line]
            num_session += line_num
            flag = False
            #print(line)
            for i in range(len(line) - window_size + 1):
                #print(i)
                if flag:
                    break
                seq = line[i:i + window_size]
                #print(seq)
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob=ExtractProb(x_decoded, input_y)
                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    for k in range(len(final_predict)):
                        if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        #if float(final_probs[k][0]) < Anomaly_Threshold:
                            flag = True
                            print("False Positive!")
                            FP1 += 1
                            FP2 += line_num
                            f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
                            ln = np.array(line)
                            for j in range(len(ln)):
                                ln[j] = int(ln[j])
                            newline = [str(m) for m in line]
                            output = ' '.join(newline)
                            f.write("{}\n".format(output))
                            f.close()
                            break
    TN = num_session - FP2
    #logging.info(f'False positive (FP): {FP1}')
    #print(f'False positive (FP): {FP1}')
    logging.info(f'False positive (FP): {FP2}')
    print(f'False positive (FP): {FP2}')
    logging.info(f'True negative (TN): {TN}')
    print(f'True negative (TN): {TN}')


    f = open(final_result, "a")  # 利用追加模式,参数从w替换为a即可
    str1 = f'False positive (FP2): {FP2}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP2}'
    f.write(str1 + "\r\n")
    # FN = test_abnormal_length - TP2
    P = 100 * TP2 / (TP2 + FP2)
    R = 100 * TP2 / (TP2 + FN)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP2, FN, P, R, F1)
    f.write(str1 + "\r\n")
    f.close()
    print('Finished Predicting')

def EvaluateAccuracy_EventLevel_Threshold(savedmodel,TopN, k):
    Anomaly_Threshold=k
    resultdir = "./UNSWNB_Results/TOPN-" + str(TopN)+"_"+str(Anomaly_Threshold)
    os.makedirs(resultdir, exist_ok=True)
    fp_file = resultdir + "/False-Positive_unswnb_MADD.txt"
    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
    f.close()
    fn_file = resultdir + "/False-negative_unswnb_MADD.txt"
    f = open(fn_file, "a")
    f.close()
    final_result = resultdir + "/statistics_unswnb_MADD.txt"
    f = open(final_result, "a")
    f.close()
    # load the model from disk
    #cluster_model = pickle.load(open("./result/unswnb/MADD_LOCALCLUSTER.pkl", 'rb'))
    savedmodel.eval()
    #for single critic
    FP1 = 0
    TP1 = 0
    #for double check
    TP2 = 0
    FP2 = 0
    window_size = 10
    num_session = 0
    total_probs_abnormal = []

    with torch.no_grad():
        for line in tqdm(test_abnormal_loader.keys()):
            # print(len(line))
            num_session += test_abnormal_loader[line]
            linenumber = test_abnormal_loader[line]
            flag = False
            flag2=False
            for i in range(len(line) - window_size + 1):
                if flag:
                    break
                seq = line[i:i + window_size]
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob = ExtractProb(x_decoded, input_y)

                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    # print(final_probs)
                    for k in range(len(final_predict)):
                        #print(float(final_probs[k][0]))
                        #if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        if float(final_probs[k][0]) < Anomaly_Threshold:
                            flag = True
                            TP1 += linenumber
                            if not flag2:
                                TP2 += linenumber
                                flag2 = True
                            break
                        if not label_x[k] in final_predict[k] and not flag2:
                            TP2 += linenumber
                            flag2 = True
                        #else:
                        #    print(float(final_probs[k][0]))
            if not flag:
                print("False Negative!")
                f = open(fn_file, "a")  # 利用追加模式,参数从w替换为a即可
                line = np.array(line)
                for i in range(len(line)):
                    line[i] = int(line[i])
                newline = [str(i) for i in line]
                output = ' '.join(newline)
                f.write("{}\n".format(output))
                f.close()
    FN2 = num_session - TP2
    FN1 = num_session - TP1
    logging.info(f'False negative (FN): {FN2}')
    print(f'False negative (FN1): {FN1}')
    print(f'False negative (FN2): {FN2}')

    num_session = 0
    total_outlier_score=[]
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            line_num = test_normal_loader[line]
            num_session += line_num
            flag = False
            flag2=False
            #print(line)
            for i in range(len(line) - window_size + 1):
                #print(i)
                if flag:
                    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
                    ln = np.array(line)
                    for j in range(len(ln)):
                        ln[j] = int(ln[j])
                    newline = [str(m) for m in line]
                    output = ' '.join(newline)
                    f.write("{}\n".format(output))
                    f.close()
                    break
                seq = line[i:i + window_size]
                #print(seq)
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob=ExtractProb(x_decoded, input_y)
                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    for k in range(len(final_predict)):
                        #if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        if float(final_probs[k][0]) < Anomaly_Threshold:
                            flag = True
                            print('False Positive-1')
                            FP1 += linenumber
                            if not flag2:
                                FP2 += linenumber
                                flag2 = True
                            break
                        if not label_x[k] in final_predict[k] and not flag2:
                            print('False Positive-2')
                            FP2 += linenumber
                            flag2 = True

    TN2 = num_session - FP2
    TN1 = num_session - FP1
    #logging.info(f'False positive (FP): {FP1}')
    #print(f'False positive (FP): {FP1}')
    logging.info(f'False positive (FP): {FP2}')
    print(f'False positive (FP): {FP2}')
    logging.info(f'True negative (TN): {TN2}')
    print(f'True negative (TN): {TN2}')
    logging.info(f'False positive (FP): {FP1}')
    print(f'False positive (FP): {FP1}')
    logging.info(f'True negative (TN): {TN1}')
    print(f'True negative (TN): {TN1}')


    f = open(final_result, "a")  # 利用追加模式,参数从w替换为a即可
    f.write("Single-threshold check results" + "\r\n")
    str1 = f'False positive (FP2): {FP1}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP1}'
    f.write(str1 + "\r\n")
    # FN = test_abnormal_length - TP2
    P = 100 * TP1 / (TP1 + FP1)
    R = 100 * TP1 / (TP1 + FN1)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP1, FN1, P, R, F1)
    f.write(str1 + "\r\n")
    f.write("Double-threshold check results" + "\r\n")
    str1 = f'False positive (FP2): {FP2}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP2}'
    f.write(str1 + "\r\n")
    # FN = test_abnormal_length - TP2
    P = 100 * TP2 / (TP2 + FP2)
    R = 100 * TP2 / (TP2 + FN2)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP2, FN2, P, R, F1)
    f.write(str1 + "\r\n")
    f.close()
    print('Finished Predicting')

def EvaluateAccuracy_EventLevel_Reconstruction(savedmodel,TopN, k):
    Anomaly_Threshold=k
    resultdir = "./UNSWNB_Results/TOPN-" + str(TopN)+"_"+str(Anomaly_Threshold)
    os.makedirs(resultdir, exist_ok=True)
    fp_file = resultdir + "/False-Positive_unswnb_MADD.txt"
    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
    f.close()
    fn_file = resultdir + "/False-negative_unswnb_MADD.txt"
    f = open(fn_file, "a")
    f.close()
    final_result = resultdir + "/statistics_unswnb_MADD.txt"
    f = open(final_result, "a")
    f.close()
    # load the model from disk
    #cluster_model = pickle.load(open("./result/unswnb/MADD_LOCALCLUSTER.pkl", 'rb'))
    savedmodel.eval()
    #for single critic
    FP1 = 0
    TP1 = 0
    #for double check
    TP2 = 0
    FP2 = 0
    window_size = 10
    num_session = 0
    total_probs_abnormal = []

    with torch.no_grad():
        for line in tqdm(test_abnormal_loader.keys()):
            # print(len(line))
            num_session += test_abnormal_loader[line]
            linenumber = test_abnormal_loader[line]
            flag = False
            flag2=False
            for i in range(len(line) - window_size + 1):
                if flag:
                    break
                seq = line[i:i + window_size]
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob = ExtractProb(x_decoded, input_y)
                # outlier test
                # outlier_predict=cluster_model.predict(latent.cpu().data.numpy())
                # outlier_score = cluster_model.decision_function(latent.cpu().data.numpy())
                # outlier_score = cluster_model.predict(latent.cpu().data.numpy())
                # outlier_score = cluster_model.score_samples(latent.cpu().data.numpy())
                # outlier_score=1000
                # total_outlier_score.append(list(outlier_score)[0])
                # outlier_score = list(outlier_score)[0]
                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    # print(final_probs)
                    for k in range(len(final_predict)):
                        # print(float(final_probs[k][0]))
                        #if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        if not label_x[k] in final_predict[k]:
                            flag = True
                            TP1 += linenumber
                            if not flag2:
                                TP2 += linenumber
                                flag2 = True
                            break
                        if float(final_probs[k][0]) < Anomaly_Threshold and not flag2:
                            TP2 += linenumber
                            flag2 = True
                        #else:
                        #    print(float(final_probs[k][0]))
            if not flag:
                print("False Negative!")
                f = open(fn_file, "a")  # 利用追加模式,参数从w替换为a即可
                line = np.array(line)
                for i in range(len(line)):
                    line[i] = int(line[i])
                newline = [str(i) for i in line]
                output = ' '.join(newline)
                f.write("{}\n".format(output))
                f.close()
    FN2 = num_session - TP2
    FN1 = num_session - TP1
    logging.info(f'False negative (FN): {FN2}')
    print(f'False negative (FN1): {FN1}')
    print(f'False negative (FN2): {FN2}')

    num_session = 0
    total_outlier_score=[]
    with torch.no_grad():
        for line in tqdm(test_normal_loader.keys()):
            line_num = test_normal_loader[line]
            num_session += line_num
            flag = False
            flag2=False
            #print(line)
            for i in range(len(line) - window_size + 1):
                #print(i)
                if flag:
                    f = open(fp_file, "a")  # 利用追加模式,参数从w替换为a即可
                    ln = np.array(line)
                    for j in range(len(ln)):
                        ln[j] = int(ln[j])
                    newline = [str(m) for m in line]
                    output = ' '.join(newline)
                    f.write("{}\n".format(output))
                    f.close()
                    break
                seq = line[i:i + window_size]
                #print(seq)
                input_x = torch.tensor(seq, dtype=torch.int).view(
                    -1, window_size).to(device)
                label_x = seq
                input_y = torch.tensor(seq[::-1], dtype=torch.int).view(
                    -1, window_size).to(device)
                input_x = input_x.squeeze(-1)
                input_y = input_y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = savedmodel.forward(input_x)
                recon_prob=ExtractProb(x_decoded, input_y)
                rec_x = x_decoded
                predicted = torch.argsort(rec_x, -1)[:, :, -TopN:]
                if AnomalyDetect_TopN:
                    final_predict = predicted.squeeze(-2).int().tolist()[::-1]
                    final_probs = recon_prob.squeeze(-2)
                    final_probs = final_probs.float().tolist()[::-1]
                    for k in range(len(final_predict)):
                        #if not label_x[k] in final_predict[k] or float(final_probs[k][0]) < Anomaly_Threshold:
                        if not label_x[k] in final_predict[k]:
                            flag = True
                            print('False Positive-1')
                            FP1 += linenumber
                            if not flag2:
                                FP2 += linenumber
                                flag2 = True
                            break
                        if float(final_probs[k][0]) < Anomaly_Threshold and not flag2:
                            print('False Positive-2')
                            FP2 += linenumber
                            flag2 = True
    TN2 = num_session - FP2
    TN1 = num_session - FP1
    #logging.info(f'False positive (FP): {FP1}')
    #print(f'False positive (FP): {FP1}')
    logging.info(f'False positive (FP): {FP2}')
    print(f'False positive (FP): {FP2}')
    logging.info(f'True negative (TN): {TN2}')
    print(f'True negative (TN): {TN2}')
    logging.info(f'False positive (FP): {FP1}')
    print(f'False positive (FP): {FP1}')
    logging.info(f'True negative (TN): {TN1}')
    print(f'True negative (TN): {TN1}')


    f = open(final_result, "a")  # 利用追加模式,参数从w替换为a即可
    f.write("Single-Reconstruction check results" + "\r\n")
    str1 = f'False positive (FP2): {FP1}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP1}'
    f.write(str1 + "\r\n")
    # FN = test_abnormal_length - TP2
    P = 100 * TP1 / (TP1 + FP1)
    R = 100 * TP1 / (TP1 + FN1)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP1, FN1, P, R, F1)
    f.write(str1 + "\r\n")
    f.write("Double-threshold check results" + "\r\n")
    str1 = f'False positive (FP2): {FP2}'
    f.write(str1 + "\r\n")
    str1 = f'True positive (TP2): {TP2}'
    f.write(str1 + "\r\n")
    # FN = test_abnormal_length - TP2
    P = 100 * TP2 / (TP2 + FP2)
    R = 100 * TP2 / (TP2 + FN2)
    F1 = 2 * P * R / (P + R)
    str1 = 'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        FP2, FN2, P, R, F1)
    f.write(str1 + "\r\n")
    f.close()
    print('Finished Predicting')


def TestCluster(savedmodel):
    X_normal_input = Unswnb_read.train_generate(name='Test_Normal')
    X_abnormal_input= Unswnb_read.train_generate(name='Test_Abnormal')
    X_normal = TensorDataset(torch.tensor(X_normal_input, dtype = torch.int).to(device),
                            torch.tensor(X_normal_input).to(device))
    X_abnormal = TensorDataset(torch.tensor(X_abnormal_input, dtype=torch.int).to(device),
                             torch.tensor(X_abnormal_input).to(device))
    X_train_input = Unswnb_read.train_generate(name='Train')
    X_train = TensorDataset(torch.tensor(X_train_input, dtype=torch.int).to(device),
                            torch.tensor(X_train_input).to(device))
    labels = []
    # z_run_normal = savedmodel.Extract_cluster(X_normal)
    # for i in range(len(z_run_normal)):
    #    labels.append('Normal')
    z_run_normal = savedmodel.Extract_cluster(X_train)
    for i in range(len(z_run_normal)):
        labels.append('Normal')
    z_run_abnormal = savedmodel.Extract_cluster(X_abnormal)
    for i in range(len(z_run_abnormal)):
        labels.append('Abnormal')
    #z_run_train = savedmodel.calculateLatentProbability(X_train)
    #for i in range(len(z_run_train)):
    #    labels.append('2')
    z_run = np.vstack((z_run_normal, z_run_abnormal))#, z_run_train))
    hex_colors = ['#008000', '#FF0000']
    #for _ in np.unique(labels):
    #    hex_colors.append('#%08X' % randint(0, 0xFFFFFF))
    colors = []
    for ii in labels:
        if ii == 'Normal':
            colors.append(hex_colors[0])
        else:
            colors.append(hex_colors[1])
    z_run_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000).fit_transform(z_run)
    plt.scatter(z_run_tsne[:len(z_run_normal), 0], z_run_tsne[:len(z_run_normal), 1], c=colors[:len(z_run_normal)],
                marker='o', label='Normal', linewidths=0)
    plt.scatter(z_run_tsne[len(z_run_normal):, 0], z_run_tsne[len(z_run_normal):, 1], c=colors[len(z_run_normal):],
                marker='o', label='Abnormal', linewidths=0)
    # plt.title('tSNE on z_run')
    plt.legend()
    plt.savefig("unswnb-tsne-cluster-trainwithabnormal.png")
    plt.show()

savedmodel = EmbedLSTMVAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            num_keys=num_keys,
            embed_dim=embed_dim,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = 0.0,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload,
            bidirectional=bidirectional,
                          options=options,
                          device=device)

savedmodel = savedmodel.to(device)
savedmodel.load_state_dict(torch.load('./result/unswnb/madd_last.pth')["state_dict"])
#X_train_input, X_train_output = Unswnb_read.train_generate(bidirectional=bidirectional)
#X_train = TensorDataset(torch.tensor(X_train_input, dtype = torch.int).to(device),
#                        torch.tensor(X_train_output).to(device))

#savedmodel.evaluate_cluster(X_train)
#Anomaly_Threshold= 0

#EvaluateAccuracy_EventLevel(savedmodel)
import math
for k in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#for k in [0.1]:
    TopN=math.ceil(num_keys*0.1)
#    print("TopnN: %.f" % TopN)
    #Anomaly_Threshold=k
    EvaluateAccuracy_EventLevel_Threshold(savedmodel,TopN,k)
#TestCluster(savedmodel)
