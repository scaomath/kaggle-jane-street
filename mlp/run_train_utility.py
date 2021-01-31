#%%
import os, sys
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.backends.cudnn.deterministic = True # for bincount

from torchsummary import summary

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from mlp import *
# %%
BATCH_SIZE = 4096
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 5
NFOLDS = 1
SCALING = 10
THRESHOLD = 0.5
SEED = 802
get_seed(SEED)

# f = np.median
# f = np.mean
f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
class UtilityLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(UtilityLoss, self).__init__()

    def forward(self, inputs, targets=resp, weight=weight, date=date, scaling=SCALING):
        '''
        inputs: prediction of the model (after sigmoid with a scaling)
        targets: resp columns
        negative of the utility for minimization
        '''
    
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # a single day
        Pi = torch.bincount(date, weight * targets * inputs)
        u = (Pi.sum()).square()/(Pi.square().sum())
        return -u
# %%
