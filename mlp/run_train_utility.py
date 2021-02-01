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

'''
Training script using a utility regularizer
'''


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
with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
#%%
# vanilla actions based on resp
train['action_0'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')
    print(f'action based on resp_{c} mean: ' ,' '*10, train['action_'+str(c)].astype(int).mean())

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']

feat_cols = [f'feature_{i}' for i in range(130)]
# %%
# feat_cols = [c for c in train.columns if 'feature' in c]
f_mean = np.mean(train[feat_cols[1:]].values, axis=0)
train.fillna(train.mean(),inplace=True)

valid = train.loc[train.date >= 450].reset_index(drop=True)
train = train.loc[train.date < 450].reset_index(drop=True)
# %%
train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

feat_cols.extend(['cross_41_42_43', 'cross_1_2'])
# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=['resp'])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=['resp'])
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
#%% sanity check
item = next(iter(train_loader))
print(item)
# %%
model = ResidualMLP(output_size=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
scheduler = None
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                                 max_lr=1e-2, epochs=EPOCHS, 
#                                                 steps_per_epoch=len(train_loader))
loss_fn = SmoothBCEwLogits(smoothing=0.01)
regularizer = UtilityLoss(alpha=1e-4, scaling=10)
es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
# %%
train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)
# %%
valid_pred = valid_epoch(model, valid_loader, device)
valid_auc = roc_auc_score(valid[target_cols].values.astype(float).reshape(-1), valid_pred)
valid_logloss = log_loss(valid[target_cols].values.astype(float).reshape(-1), valid_pred)
valid_pred = valid_pred.reshape(-1, len(target_cols))
# valid_pred = f(valid_pred[...,:len(target_cols)], axis=-1) # only do first 5
valid_pred = f(valid_pred, axis=-1) # all
valid_pred = np.where(valid_pred >= THRESHOLD, 1, 0).astype(int)
valid_score = utility_score_bincount(date=valid.date.values, 
                                    weight=valid.weight.values,
                                    resp=valid.resp.values, 
                                    action=valid_pred)
# %%
train_loss = train_epoch_utility(model, optimizer, scheduler, loss_fn, regularizer, train_loader, device)
# %%
