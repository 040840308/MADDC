import numpy as np
import torch
import gc
from torch import nn, optim
from torch import distributions
from base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import math
import time
import torch.nn.functional as F
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from pyod.models.knn import KNN
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'
                 ,bidirectional=True):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.bidirectional = bidirectional

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, num_layers=self.hidden_layer_depth, dropout = dropout
                                 ,bidirectional=self.bidirectional)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout
                                ,bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return F.relu(h_end)


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
            #return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length,
                 embed_dim,output_size, dtype, block='LSTM',bidirectional=True):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.embed_dim=embed_dim
        self.output_size = output_size
        self.dtype = dtype

        self.bidirectional = bidirectional

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, num_layers=self.hidden_layer_depth, bidirectional=self.bidirectional)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth, bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_embed = nn.Linear(self.hidden_size*self.num_directions, self.embed_dim)#backward
        self.embed_to_output = nn.Linear(self.embed_dim, self.output_size)
        self.hidden_to_output=nn.Linear(self.hidden_size*self.num_directions, self.output_size)


        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth*self.num_directions, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        self.finalout=nn.Sequential(
            self.hidden_to_embed,
            nn.ReLU(),
            self.embed_to_output
        )

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_embed.weight)
        nn.init.xavier_uniform_(self.embed_to_output.weight)
        #nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = F.relu(self.latent_to_hidden(latent))
        #h_state=self.latentout(latent)
        # check whether to predict one example,
        if h_state.size(0) != self.batch_size:
            self.decoder_inputs = torch.zeros(self.sequence_length, h_state.size(0), 1, requires_grad=True).type(
                self.dtype)
            self.c_0 = torch.zeros(self.num_directions*self.hidden_layer_depth, h_state.size(0), self.hidden_size, requires_grad=True).type(
                self.dtype)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.num_directions*self.hidden_layer_depth)])
            decoder_output, (h_n, c_n) = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.num_directions*self.hidden_layer_depth)])
            decoder_output, (h_n, c_n) = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out=self.finalout(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class Embeddings(nn.Module):
    def __init__(self, embed_dim, num_keys):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(num_keys, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        #print(x.shape)
        #print(x)
        emb = self.lut(x) * math.sqrt(self.embed_dim)
        return emb

class EmbedLSTMVAE(BaseEstimator, nn.Module):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param block: GRU/LSTM to be used as a basic building block
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param optimizer: ADAM/ SGD optimizer to reduce the loss function
    :param loss: SmoothL1Loss / MSELoss / ReconLoss / any custom loss which inherits from `_Loss` class
    :param boolean cuda: to be run on GPU or not
    :param print_every: The number of iterations after which loss should be printed
    :param boolean clip: Gradient clipping to overcome explosion
    :param max_grad_norm: The grad-norm to be clipped
    :param dload: Download directory where models are to be dumped
    """
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',num_keys=29, embed_dim=512,
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=True, print_every=100, lr_step=None, lr_decay_ratio=0, clip=True,
                 max_grad_norm=6, dload='.', bidirectional=False, options=None, device=None):

        super(EmbedLSTMVAE, self).__init__()
        print("start to reconstruct!")
        self.dtype = torch.FloatTensor
        self.use_cuda = cuda
        self.embed_dim=embed_dim
        self.num_keys=num_keys
        self.KL_Weight=1
        self.REC_Weight=1
        self.lr_step=lr_step
        self.lr_decay_ratio=lr_decay_ratio
        self.options=options
        self.save_dir=options['save_dir']
        self.total_latent=[]

        os.makedirs(self.save_dir, exist_ok=True)

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor

        self.src_embed=Embeddings(embed_dim=self.embed_dim,num_keys=self.num_keys)


        self.encoder = Encoder(number_of_features = embed_dim,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block,
                               bidirectional=bidirectional)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               embed_dim=embed_dim,
                               output_size=num_keys,
                               block=block,
                               dtype=self.dtype,
                               bidirectional=bidirectional)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = self.save_dir

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device=device

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)
        elif loss == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss(size_average=False)

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.state_dict(),
            "log": self.log,
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + 'madd' + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def save_parameters(self,options, filename):
        with open(filename, "w") as f:
            for key in options.keys():
                f.write("{}: {}\n".format(key, options[key]))

    def __repr__(self):
        #print("start to VRAE")
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, X):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        #print("start to forward!")
        embed_x=self.src_embed(X.to(self.device))
        #
        embed_x=embed_x.permute(1,0,2)
        cell_output = self.encoder(embed_x.to(self.device))
        latent = self.lmbd(cell_output.to(self.device))
        x_decoded= self.decoder(latent.to(self.device))

        reconstruct_probability=[0]

        return x_decoded, latent, reconstruct_probability

    def forwardForcluster(self, X, Y):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x:input tensor
        :return: the decoded output, latent vector
        """
        #print("start to forward!")
        embed_x=self.src_embed(X.to(self.device))
        #
        embed_x = embed_x.permute(1, 0, 2)
        cell_output = self.encoder(embed_x.to(self.device))
        latent = self.lmbd(cell_output.to(self.device))
        x_decoded = self.decoder(latent.to(self.device))

        # extract probability
        softmax_out = F.softmax(x_decoded, dim=-1)
        y = Y.tolist()
        softmax_out = softmax_out.tolist()
        df_softmax = pd.DataFrame(softmax_out)
        df_y = pd.DataFrame(y)
        df = df_softmax + df_y
        df = df.applymap(lambda x: x[int(x[self.num_keys])])  # num_keys is to extract the index probability, 1+num_keys-1
        prob = np.array(df)
        reconstruct_probability = torch.tensor(prob)
        reconstruct_probability = reconstruct_probability.unsqueeze(-1)

        return x_decoded, latent, reconstruct_probability

    def _rec(self, x_decoded, x, loss_fn):
        """
        Compute the loss given output x decoded, input x and the specified loss function

        :param x_decoded: output of the decoder
        :param x: input to the encoder
        :param loss_fn: loss function specified
        :return: joint loss, reconstruction loss and kl-divergence loss
        """
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss= -0.5* torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        rec_x=x_decoded.contiguous().view(-1, x_decoded.size(-1))
        label=x.contiguous().view(-1)
        label = Variable(label[:].type(torch.int64), requires_grad=False)

        recon_loss = loss_fn(rec_x, label)
        loss=self.REC_Weight*recon_loss+self.KL_Weight*kl_loss

        return loss, recon_loss, kl_loss

    def compute_loss(self, X, Y):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration

        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        Y = Y.permute(1, 0).unsqueeze(-1)
        x_decoded, latent, recon_prob = self.forward(X.to(self.device))
        loss, recon_loss, kl_loss = self._rec(x_decoded, Y, self.loss_fn)

        return loss, recon_loss, kl_loss, X, latent, recon_prob


    def _train(self, train_loader):
        """
        For each epoch, given the batch_size, run this function batch_size * num_of_batches number of times

        :param train_loader:input train loader with shuffle
        :return:
        """
        self.train()
        gc.collect()
        epoch_loss = 0
        REC_loss=0
        average_rec_loss=0
        KL_loss=0
        average_kl_loss=0
        t = 0
        num_batch = len(train_loader)

        for t, (X,Y) in enumerate(train_loader):

            # Index first element of array to return tensor
            X = X
            Label = Y
            #print(X)
            #print(Y)
            #for hdfs-old, to embed need squeeze
            # required to swap axes, since dataloader gives output in (batch_size x seq_len x num_of_features)
            #X = X.permute(1,0,2)
            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, X, latent, recon_prob= self.compute_loss(X.to(self.device),Label)
            #if recon_loss > 5.0 and kl_loss >7.0:
            #    loss.backward()
            #elif recon_loss <2.0 and kl_loss>4.0:
            #    kl_loss.backward()
            #elif recon_loss >1.0 and kl_loss<2.0:
            #    recon_loss.backward()
            #else:
            #    loss.backward()
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()
            REC_loss += recon_loss.item()
            KL_loss +=kl_loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))

        average_rec_loss = REC_loss / t
        average_kl_loss = KL_loss / t
        print('Average loss: {:.4f} Average Recon loss: {:.4f} Average KL loss: {:.4f}'.format(epoch_loss / t,
                                                                                               average_rec_loss, average_kl_loss))
        self.log['train']['loss'].append(epoch_loss / num_batch)



    def fit(self, dataset, save = False):
        """
        Calls `_train` function over a fixed number of epochs, specified by `n_epochs`

        :param dataset: `Dataset` object
        :param bool save: If true, dumps the trained model parameters as pickle file at `dload` directory
        :return:
        """

        train_loader = DataLoader(dataset = dataset,
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  drop_last=True)


        self.save_parameters(self.options,self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

        for i in range(self.n_epochs):
            print('Epoch: %s' % i)
            if i  == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if i  in [1,2,3,4,5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if i  in self.lr_step:
                if i <100:
                    self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
                else:
                    self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
                #self.KL_Weight *=1.5
            self.log['train']['epoch'].append(i)
            start = time.strftime("%H:%M:%S")
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log['train']['lr'].append(lr)
            self.log['train']['time'].append(start)

            self._train(train_loader)

            if save and i %10==0:
                self.save('madd_model.pth')
                self.save_checkpoint(i,
                                     save_optimizer=True,
                                     suffix="epoch" + str(i))
            self.save_checkpoint(i, save_optimizer=True, suffix="last")
            self.save_log()

        self.is_fitted = True
        if save:
            self.save('madd_model.pth')


    def _batch_transform(self, x):
        """
        Passes the given input tensor into encoder and lambda function

        :param x: input batch tensor
        :return: intermediate latent vector
        """
        #x=x.squeeze(-1)
        embed_x=self.src_embed(x)
        #
        embed_x = embed_x.permute(1, 0, 2)
        x=embed_x.permute(1,0,2)
        return self.lmbd(
                    self.encoder(
                        Variable(x, requires_grad = False)
                    )
        ).cpu().data.numpy()
        #return self.encoder(
        #               Variable(x, requires_grad = False)
        #            ).cpu().data.numpy()

    def _batch_reconstruct(self, x):
        """
        Passes the given input tensor into encoder, lambda and decoder function

        :param x: input batch tensor
        :return: reconstructed output tensor
        """

        x = Variable(x.type(self.dtype), requires_grad = False)
        x_decoded, latent= self.forward(x)

        return x_decoded.cpu().data.numpy(), latent.cpu().data.numpy()

    def reconstruct(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_reconstruct`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's output vectors are to be obtained
        :param bool save: If true, dumps the output vector dataframe as a pickle file
        :return:
        """

        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader

        if self.is_fitted:
            with torch.no_grad():
                x_decoded = []

                for t, (x,y) in enumerate(test_loader):
                    x = x[0]
                    x = x.permute(1, 0, 2)

                    x_decoded_each = self._batch_reconstruct(x)
                    x_decoded.append(x_decoded_each)

                x_decoded = np.concatenate(x_decoded, axis=1)

                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    x_decoded.dump(self.dload + '/z_run.pkl')
                return x_decoded

        raise RuntimeError('Model needs to be fit')

    def evaluate_cluster(self,dataset):
        self.eval()

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 drop_last=True)  # Don't shuffle for test_loader
        with torch.no_grad():
            z_run = []

            for t, (x,y) in enumerate(data_loader):
                # print("start to forward!")
                embed_x = self.src_embed(x.to(self.device))
                #
                embed_x = embed_x.permute(1, 0, 2)
                cell_output = self.encoder(embed_x.to(self.device))
                latent = self.lmbd(cell_output.to(self.device))
                #z_run_each = self._batch_transform(x)
                z_run.append(latent)
            z_run = np.concatenate(z_run, axis=0)

            #import numpy as np
            #print(np.mean(z))
            #return z_run

            CLUSTERMODEL_PATH = self.dload + '/' + "MADD_LOCALCLUSTER.pkl"
            #rng = np.random.RandomState(42)
            # fit the model
            #lof = IsolationForest(max_samples='auto', random_state=rng)
            #lof=LocalOutlierFactor(n_neighbors=50,algorithm='auto',metric='euclidean',
            #                       contamination=0.02,novelty=True)
            lof = EllipticEnvelope(contamination=0.008)
            #lof=OneClassSVM(kernel='rbf', nu=0.2, tol=1e-3,
            #                shrinking=True, gamma=0.1, max_iter=-1)
            lof.fit(z_run)
            import pickle
            with open(CLUSTERMODEL_PATH,'wb') as f:
                print("Save the Local Cluster Model!")
                pickle.dump(lof,f)
            # Continuous output of the decision_function
            decision = lof.decision_function(z_run)
            import scipy.stats as stats
            # Get the "thresholding" value from the decision function
            threshold = stats.scoreatpercentile(lof.decision_function(z_run), 100 * 0.003)
            print(decision.min())
            print(decision.max())
            print(threshold)
            score_samples=lof.score_samples(z_run)
            print(score_samples.min())
            print(score_samples.max())
            threshold=stats.scoreatpercentile(lof.score_samples(z_run), 100 * 0.003)
            print(threshold)

    def Extract_cluster(self,dataset):
        self.eval()
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 drop_last=True)  # Don't shuffle for test_loader
        with torch.no_grad():
            z_run = []
            for t, (x,y) in enumerate(data_loader):
                # print("start to forward!")
                embed_x = self.src_embed(x.to(self.device))
                #
                embed_x = embed_x.permute(1, 0, 2)
                cell_output = self.encoder(embed_x.to(self.device))
                #latent = self.lmbd(cell_output.to(self.device))
                #z_run_each = self._batch_transform(x)
                z_run.append(cell_output.cpu().data.numpy())
            z_run = np.concatenate(z_run, axis=0)
            return z_run

    def ReconstructProbabilityCluster(self,dataset):
        self.eval()
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 drop_last=True)  # Don't shuffle for test_loader
        with torch.no_grad():
            x_decoded = []
            prob=[]
            latent=[]
            from tqdm import tqdm
            for t, (x,y) in tqdm(enumerate(test_loader)):
                X = x
                y= y.permute(1, 0).unsqueeze(-1)
                x_decoded, latent, recon_prob = self.forwardForcluster(X,y)
                torch.cuda.empty_cache()
                prob.append(recon_prob)

            prob = np.concatenate(prob, axis=0)
            #start to train prob model for outlier detection
            #isolation forest
            rng = np.random.RandomState(42)
            outlier_model= IsolationForest(random_state=rng)
            # fit the model
            outlier_model.fit(prob)
            CLUSTERMODEL_PATH = self.dload + '/' + "Prob_IsolationForest.pkl"
            import pickle
            with open(CLUSTERMODEL_PATH, 'wb') as f:
                print("Save the Local Cluster Model!")
                pickle.dump(outlier_model, f)

            # LOF
            outlier_model = LocalOutlierFactor(novelty=True)
            # fit the model
            outlier_model.fit(prob)
            CLUSTERMODEL_PATH = self.dload + '/' + "Prob_LOF.pkl"
            import pickle
            with open(CLUSTERMODEL_PATH, 'wb') as f:
                print("Save the Local Cluster Model!")
                pickle.dump(outlier_model, f)
            #EllipticEnvelope
            outlier_model = EllipticEnvelope()
            # fit the model
            outlier_model.fit(prob)
            CLUSTERMODEL_PATH = self.dload + '/' + "Prob_EE.pkl"
            import pickle
            with open(CLUSTERMODEL_PATH, 'wb') as f:
                print("Save the Local Cluster Model!")
                pickle.dump(outlier_model, f)
            #OneClassSVM
            outlier_model = OneClassSVM()
            # fit the model
            outlier_model.fit(prob)
            CLUSTERMODEL_PATH = self.dload + '/' + "Prob_OneSVM.pkl"
            import pickle
            with open(CLUSTERMODEL_PATH, 'wb') as f:
                print("Save the Local Cluster Model!")
                pickle.dump(outlier_model, f)

    def transform(self, dataset, save = False):
        """
        Given input dataset, creates dataloader, runs dataloader on `_batch_transform`
        Prerequisite is that model has to be fit

        :param dataset: input dataset who's latent vectors are to be obtained
        :param bool save: If true, dumps the latent vector dataframe as a pickle file
        :return:
        """
        self.eval()

        test_loader = DataLoader(dataset = dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 drop_last=True) # Don't shuffle for test_loader
        if self.is_fitted:
            with torch.no_grad():
                z_run = []

                for t, (x,y) in enumerate(test_loader):
                    #x = x[0]
                    #x = x.permute(1, 0, 2)

                    z_run_each = self._batch_transform(x)
                    z_run.append(z_run_each)

                z_run = np.concatenate(z_run, axis=0)
                if save:
                    if os.path.exists(self.dload):
                        pass
                    else:
                        os.mkdir(self.dload)
                    z_run.dump(self.dload + '/z_run.pkl')
                return z_run

        raise RuntimeError('Model needs to be fit')

    def fit_transform(self, dataset, save = False):
        """
        Combines the `fit` and `transform` functions above

        :param dataset: Dataset on which fit and transform have to be performed
        :param bool save: If true, dumps the model and latent vectors as pickle file
        :return: latent vectors for input dataset
        """
        self.fit(dataset, save = save)
        return self.transform(dataset, save = save)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later

        :param file_name: the filename to be saved as,`dload` serves as the download directory
        :return: None
        """
        PATH = self.dload + '/' + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned

        :param PATH: Should contain pickle file
        :return: None
        """
        self.is_fitted = True
        self.load_state_dict(torch.load(PATH))