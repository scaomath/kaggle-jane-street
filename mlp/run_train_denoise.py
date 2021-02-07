# %%
from utils import *
from mlp import *
from torchsummary import summary
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
torch.backends.cudnn.deterministic = True  # for bincount

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)

# %%

'''
Training script finetuning using resp colums as regularizer with an additional denoised target
from https://www.kaggle.com/lucasmorin/target-engineering-patterns-denoising
'''

DEBUG = False
LOAD_PRETRAIN = False
TRAINING_START = 86  # 86 by default
FINETUNE_BATCH_SIZE = 2048_00
BATCH_SIZE = 8196
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 6
NFOLDS = 1
SCALING = 10
THRESHOLD = 0.5

SEED = 1127802
get_seed(SEED)

# f = np.median
# f = np.mean
f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train, valid = preprocess_pt(train_parquet, day_start=TRAINING_START, drop_zero_weight=False)



print(f'action based on resp mean:   ', train['action_0'].astype(int).mean())
for c in range(1, 5):
    print(f'action based on resp_{c} mean: ',
          train['action_'+str(c)].astype(int).mean())

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
resp_cols_all = resp_cols
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
feat_cols = [f'feature_{i}' for i in range(130)]
# f_mean = np.mean(train[feat_cols[1:]].values, axis=0)
feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

# %%
train_set = ExtendedMarketDataset(
    train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(
    valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = ResidualMLP(hidden_size=256, output_size=len(target_cols))
# model = MLP(hidden_units=(None,160,160,160), input_dim=len(feat_cols), output_dim=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

# %%
