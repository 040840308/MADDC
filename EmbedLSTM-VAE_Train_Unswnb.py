from EmbedLSTMVAE import EmbedLSTMVAE
import utils
import Unswnb_read
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

#os.makedirs('test',exist_ok=True)

dload='./model_dir'
#hyper parameters
options={}
#hyper parameters
hidden_size = 128
options['hidden_size']=hidden_size
hidden_layer_depth = 2
options['hidden_layer_depth']=hidden_layer_depth
latent_length = 30
options['latent_length']=latent_length
batch_size = 2048
options['batch_size']=batch_size
learning_rate = 0.001#modify default to 0.005,0.005, rec 2, kl 8
options['learning_rate']=learning_rate
lr_step = (60,80)
options['lr_step']=lr_step
lr_decay_ratio = 0.3
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
max_grad_norm=10
options['max_grad_norm']=max_grad_norm
loss = 'CrossEntropyLoss' # options: SmoothL1Loss, MSELoss
options['loss']=loss
block = 'LSTM' # options: LSTM, GRU
options['block']=block
options['save_dir']="./result/unswnb/"

train_file=''

bidirectional=False

X_train_input, X_train_output = Unswnb_read.train_generate(bidirectional=bidirectional)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = TensorDataset(torch.tensor(X_train_input, dtype = torch.int).to(device),
                        torch.tensor(X_train_output).to(device))

sequence_length = 10
number_of_features = 1

vrae = EmbedLSTMVAE(sequence_length=sequence_length,
            number_of_features = number_of_features,
            num_keys=num_keys,
            embed_dim=embed_dim,
            hidden_size = hidden_size,
            hidden_layer_depth = hidden_layer_depth,
            latent_length = latent_length,
            batch_size = batch_size,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            dropout_rate = dropout_rate,
            optimizer = optimizer,
            cuda = cuda,
            print_every=print_every,
            lr_step=lr_step,
            lr_decay_ratio=lr_decay_ratio,
            clip=clip,
            max_grad_norm=max_grad_norm,
            loss = loss,
            block = block,
            dload = dload,
            bidirectional=bidirectional,
            options=options)

#If the model has to be saved, with the learnt parameters use:
vrae.fit(X_train, save = True)
vrae.save('embedLSTMvae-Unswnb.pth')
#vrae.evaluate_cluster(X_train)
