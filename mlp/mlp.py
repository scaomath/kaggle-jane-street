#%%
import warnings
warnings.filterwarnings('ignore')

import datetime
import os
import sys
import warnings
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

CURRENT = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(CURRENT)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from utils_js import *

#%%
NFOLDS = 5

feat_cols = [f'feature_{i}' for i in range(130)]
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']

hidden_units = [None, 160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]

f_mean = np.load(os.path.join(DATA_DIR,'f_mean.npy'))

##### Making features
all_feat_cols = [col for col in feat_cols]
all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

val_util_thresh = 5000

##### Model&Data fnc
class ResidualMLP(nn.Module):
    def __init__(self, hidden_size=256, 
                       output_size=len(target_cols), 
                       input_size=len(all_feat_cols),
                       dropout_rate=0.2):
        super(ResidualMLP, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.dropout0 = nn.Dropout(0.2)

        self.dense1 = nn.Linear(len(all_feat_cols), hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+len(all_feat_cols), hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.dense5 = nn.Linear(hidden_size+hidden_size, output_size)

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x4 = self.dense4(x)
        x4 = self.batch_norm4(x4)
        # x = F.relu(x)
        # x = self.PReLU(x)
        x4 = self.LeakyReLU(x4)
        x4 = self.dropout4(x4)

        x = torch.cat([x3, x4], 1)

        x = self.dense5(x)

        return x


class MLP(nn.Module):
    def __init__(self, hidden_units=hidden_units,
                       dropout_rates=dropout_rates,
                       input_dim=len(all_feat_cols),
                       output_dim=len(target_cols)):
        super(MLP, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(input_dim)
        self.dropout0 = nn.Dropout(dropout_rates[0])
        
        self.dense1 = nn.Linear(input_dim, hidden_units[1])
        self.batch_norm1 = nn.BatchNorm1d(hidden_units[1]) 
        self.dropout1 = nn.Dropout(dropout_rates[1])
        
        self.dense2 = nn.Linear(hidden_units[1], hidden_units[2])
        self.batch_norm2 = nn.BatchNorm1d(hidden_units[2])
        self.dropout2 = nn.Dropout(dropout_rates[2])
        
        self.dense3 = nn.Linear(hidden_units[2], hidden_units[3])
        self.batch_norm3 = nn.BatchNorm1d(hidden_units[3])
        self.dropout3 = nn.Dropout(dropout_rates[3])
        
        self.dense4 = nn.Linear(hidden_units[3], output_dim)
        
        self.silu = nn.SiLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)
        
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.silu(x)
        x = self.dropout1(x)
    
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.silu(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.silu(x)
        x = self.dropout3(x)
        
        x = self.dense4(x)
        
        return x

class RunningEWMean:
    def __init__(self, WIN_SIZE=20, n_size=1, lt_mean=None):
        if lt_mean is not None:
            self.s = lt_mean
        else:
            self.s = np.zeros(n_size)
        self.past_value = np.zeros(n_size)
        self.alpha = 2 / (WIN_SIZE + 1)

    def clear(self):
        self.s = 0

    def push(self, x):
        x = fast_fillna(x, self.past_value)
        self.past_value = x
        self.s = self.alpha * x + (1 - self.alpha) * self.s

    def get_mean(self):
        return self.s

class MarketDataset:
    def __init__(self, df, 
                       features=all_feat_cols, 
                       targets=target_cols):
        self.features = df[features].values
        self.label = df[targets].astype('float').values.reshape(-1, len(targets))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }

class ExtendedMarketDataset:
    def __init__(self, df, features=feat_cols, 
                           targets=target_cols,
                           resp = resp_cols,
                           date='date',
                           weight='weight'):
        self.features = df[features].values
        self.label = df[targets].astype('int').values.reshape(-1, len(targets))
        self.resp = df[resp].astype('float').values.reshape(-1, len(resp))
        self.date = df[date].astype('int').values.reshape(-1,1)
        self.weight = df[weight].astype('float').values.reshape(-1,1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
            'resp':torch.tensor(self.resp[idx], dtype=torch.float),
            'date':torch.tensor(self.date[idx], dtype=torch.int32),
            'weight':torch.tensor(self.weight[idx], dtype=torch.float),
        }

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class RespMSELoss(nn.Module):
    def __init__(self, weight=None, alpha=None, reduction='mean', scaling=None, resp_index=None):
        super(RespMSELoss, self).__init__()
        self.alpha = alpha # the strength of this regularization
        self.scaling = scaling
        self.reduction = reduction
        self.weight = weight
        self.resp_index = resp_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, weight=None, date=None):
        '''
        inputs: prediction of the model (without sigmoid, processed with a scaling)
        targets: resp columns
        simple MSE for resp
        date is just for implementation uniformity
        '''
        if self.resp_index is not None and len(self.resp_index) < 5:
            inputs = inputs[..., self.resp_index]
            targets = targets[..., self.resp_index]
        
        if self.scaling is not None:
            inputs *= self.scaling

        n_targets = inputs.size(-1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        loss = (inputs - targets).square()

        if weight is not None:
            if n_targets > 1:
                weight = weight.repeat((n_targets,1))
            weight = weight.view(-1)
            loss *= weight

        if self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss.to(self.device)

class UtilityLoss(nn.Module):
    def __init__(self, weight=None, alpha=None, scaling=None, normalize='mean', resp_index=None):
        super(UtilityLoss, self).__init__()
        self.alpha = alpha if normalize == 'mean' else alpha * \
            1e-3  # the strength of this regularization
        self.normalize = normalize
        self.scaling = scaling
        self.weight = weight
        self.resp_index = resp_index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, weights=None, date=None):
        '''
        inputs: prediction of the model (without sigmoid, processed with a scaling)
        targets: resp columns
        negative of the utility for minimization
        '''
        if (self.resp_index is not None) and (len(self.resp_index) < 5):
            inputs = inputs[..., self.resp_index]
            targets = targets[..., self.resp_index]

        inputs = F.sigmoid(self.scaling*inputs)
        n_targets = len(self.resp_index)
        if n_targets > 1:
            weights = weights.repeat((n_targets, 1))
            date = date.repeat((n_targets, 1))

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weights = weights.view(-1)
        date = date.view(-1)

        dates = date.unique().detach()
        ndays = len(dates)

        Pi = torch.zeros((ndays, 1), device=self.device, dtype=torch.float32)
        for i, day in enumerate(dates):
            mask = (date == day)
            Pi[i] = (weights[mask]*targets[mask]*inputs[mask]).sum()

        # a single day
        # DEBUG notes: bincount is not differentiable for autograd
        # Pi = torch.bincount(date, weight * targets * inputs)
        # loss = Pi.sum()*(Pi.sum().clamp(min=0))/(Pi.square().sum())
        # loss = (Pi.sum()).square()/(Pi.square().sum())

        sumPi = Pi.sum()
        if self.normalize == 'mean':
            loss = -self.alpha*sumPi * \
                (sumPi.clamp(min=0))/(Pi.square().sum())/ndays
        else:
            loss = -self.alpha*sumPi*(sumPi.clamp(min=0))/ndays

        return loss

#Designed to do all features at the same time, but Kaggle kernels are memory limited.
class NeutralizeTransform:
    def __init__(self,proportion=1.0):
        self.proportion = proportion
    
    def fit(self,X,y):
        self.lms = []
        self.mean_exposure = np.mean(y,axis=0)
        self.y_shape = y.shape[-1]
        for x in X.T:
            scores = x.reshape((-1,1))
            exposures = y
            exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))
            
            transform = np.linalg.lstsq(exposures, scores, rcond=None)[0]
            self.lms.append(transform)
            
    def transform(self,X,y=None):
        out = []
        for i,transform in enumerate(self.lms):
            x = X[:,i]
            scores = x.reshape((-1,1))
            exposures = np.repeat(self.mean_exposure,len(x),axis=0).reshape((-1,self.y_shape))
            exposures = np.concatenate([exposures,np.array([np.mean(scores)] * len(exposures)).reshape((-1,1))],axis=1)
            correction = self.proportion * exposures.dot(transform)
            out.append(x - correction.ravel())
            
        return np.asarray(out).T
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X,y)

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0, monitor='utility', save_threshold=5000):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.monitor = monitor
        self.best_score = None
        self.best_utility_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.message = None
        self.save_threshold = save_threshold

    def __call__(self, epoch_score, model, model_path, epoch_utility_score):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        util_score = epoch_utility_score

        if self.monitor == 'utility':
            if self.best_utility_score is None:
                self.best_utility_score = util_score
            elif util_score < self.best_utility_score:
                self.counter += 1
                self.message = f'EarlyStopping counter: {self.counter} out of {self.patience}'
                if self.counter >= self.patience: # a harder offset
                    self.early_stop = True
            else:
                self.message = f'Utility score :({self.best_utility_score:.2f} --> {util_score:.2f}).'
                self.best_utility_score = util_score
                if util_score > self.save_threshold:
                    self.message += " model saved."
                    self.save_checkpoint(epoch_score, model, model_path)
                self.counter = 0
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(epoch_score, model, model_path)
            elif score < self.best_score: #  + self.delta
                self.counter += 1
                self.message = f'EarlyStopping counter: {self.counter} out of {self.patience}'
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.message = f'Valid score :({self.best_score:.4f} --> {score:.4f}).'
                self.best_score = score
                if score > self.save_threshold:
                    self.message += " model saved."
                    self.save_checkpoint(epoch_score, model, model_path)
                self.counter = 0

        

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            torch.save(model.state_dict(), model_path)
            _ = f'Validation score improved ({self.val_score} --> {epoch_score})'
        self.val_score = epoch_score




def train_epoch(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0
    len_data = len(dataloader)

    with tqdm(total=len_data) as pbar:
        for data in dataloader:
            optimizer.zero_grad()
            features = data['features'].to(device)
            label = data['label'].to(device)
            outputs = model(features)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            final_loss += loss.item()
            pbar.update()

    final_loss /= len(dataloader)

    return final_loss

def train_epoch_utility(model, optimizer, scheduler, regularizer, dataloader, device, loss_fn=None):
    model.train()
    # final_loss = 0
    util_loss = 0
    final_loss = 0
    len_data = len(dataloader)
    _loss = None

    with tqdm(total=len_data) as pbar:
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            features = data['features'].to(device)
            label = data['label'].to(device)
            weight = data['weight'].to(device)
            resp = data['resp'].to(device)
            date = data['date'].to(device)
            outputs = model(features)
            loss = regularizer(outputs, resp, weights=weight, date=date)
            util_loss += -loss.item()
            if loss_fn is not None:
                _loss = loss_fn(outputs, label)
                loss += _loss
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # with torch.no_grad():
            #     _loss = loss_fn(outputs, label)
            
            if _loss is not None:
                final_loss += _loss.item()
                pbar.set_description(f"Train loss: {final_loss/(i+1):.8f} \t Train utility: {util_loss/(i+1):.4e}")
            else:
                pbar.set_description(f"Utility regularizer val: {util_loss/(i+1):.4e}")
            pbar.update()

    return util_loss

def train_epoch_finetune(model, optimizer, scheduler, regularizer, dataloader, device, 
                         loss_fn=None, weighted=True):
    model.train()
    utils_loss = 0
    train_loss = 0
    len_data = len(dataloader)
    _loss = None

    with tqdm(total=len_data) as pbar:
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            features = data['features'].to(device)
            label = data['label'].to(device)
            weight = data['weight'].to(device)
            resp = data['resp'].to(device)
            date = data['date'].to(device)
            outputs = model(features)
            loss = regularizer(outputs, resp, weights=weight, date=date)

            if loss.item() < 0:
                utils_loss += -loss.item()
            else:
                utils_loss += loss.item()

            if loss_fn is not None:
                _loss = loss_fn(outputs, label)
                loss += _loss

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # with torch.no_grad():
            #     _loss = loss_fn(outputs, label)
            
            if _loss is not None:
                train_loss += _loss.item()
                desc = f"Train loss: {train_loss/(i+1):.8f} \t" 
                desc += f"Fine-tuning {regularizer.__class__.__name__} loss: {utils_loss/(i+1):.4e}"
                pbar.set_description(desc)
            else:
                desc = f"Fine-tuning {regularizer.__class__.__name__} loss: {utils_loss/(i+1):.4e}"
                pbar.set_description(desc)
            pbar.update()

    return utils_loss

def valid_epoch(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)

    return preds

def get_valid_score(preds, valid_df, f=median_avg, threshold=0.5,target_cols=target_cols):
    valid_auc = roc_auc_score(valid_df[target_cols].values.astype(float).reshape(-1), preds)
    valid_logloss = log_loss(valid_df[target_cols].values.astype(float).reshape(-1), preds)
    valid_pred = preds.reshape(-1, len(target_cols))
    # valid_pred = f(valid_pred[...,:len(target_cols)], axis=-1) # only do first 5
    valid_pred = f(valid_pred, axis=-1) # all
    valid_pred = np.where(valid_pred >= threshold, 1, 0).astype(int)
    valid_score = utility_score_bincount(date=valid_df.date.values, 
                                        weight=valid_df.weight.values,
                                        resp=valid_df.resp.values, 
                                        action=valid_pred)
    return valid_auc, valid_score
#%%
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = ResidualMLP()
    model.to(device)
    n_params = get_num_params(model)
    print(f'Number of params: {n_params}')

    try:
        from torchsummary import summary
        summary(model, input_size=(len(all_feat_cols), ))
    except ImportError as e:
        print(f"{str(datetime.datetime.now())} Import error {e}")

