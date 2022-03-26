from EmbedLSTMVAE import EmbedLSTMVAE
import utils
import numpy as np
import torch
from sklearn.manifold import TSNE
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
dload='./model_dir'
#hyper parameters
options={}
#hyper parameters
hidden_size = 128
options['hidden_size']=hidden_size
hidden_layer_depth = 2
options['hidden_layer_depth']=hidden_layer_depth
latent_length = 32
options['latent_length']=latent_length
batch_size = 1024
options['batch_size']=batch_size
learning_rate = 0.001#modify default to 0.005,0.005, rec 2, kl 8
options['learning_rate']=learning_rate
lr_step = (60, 80)
options['lr_step']=lr_step
lr_decay_ratio = 0.2
options['lr_decay_ration']=lr_decay_ratio
n_epochs = 100
options['n_epochs']=n_epochs
num_keys=32
options['num_keys']=num_keys
embed_dim=64
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
options['save_dir']="./result/deeplog/"

train_file='myhdfs/hdfs_train'

bidirectional=False

X_train_input, X_train_output = utils.hdfs_generate(bidirectional=bidirectional)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = TensorDataset(torch.as_tensor(X_train_input, dtype = torch.int).to(device),
                        torch.as_tensor(X_train_output).to(device))

sequence_length = 10
number_of_features = 1

vae_model = EmbedLSTMVAE(sequence_length=sequence_length,
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
vae_model.fit(X_train, save = True)
vae_model.save('embedLSTMvae.pth')
