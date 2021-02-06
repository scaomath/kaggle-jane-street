#%%
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.backends.cudnn.deterministic = True  # for bincount

from torchsummary import summary

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)
from mlp import *
from utils import *

#%%
feat_cols = [f'feature_{i}' for i in range(130)]
all_feat_cols = feat_cols
all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])
# model parameters
hidden_units = [None, 160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
num_columns = len(all_feat_cols)
num_labels = len(target_cols)
# %%
class MLPModel(nn.Module):
    
    
    def __init__(self, hidden_units=hidden_units,
                       dropout_rates=dropout_rates,
                       input_dim=num_columns,
                       output_dim=num_labels):
        super(MLPModel, self).__init__()
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