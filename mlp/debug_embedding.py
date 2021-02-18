#%%
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)
# for f in ['/home/scao/anaconda3/lib/python3.8/lib-dynload', 
#           '/home/scao/anaconda3/lib/python3.8/site-packages']:
#     sys.path.append(f) 

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from utils import *
from utils_js import *
from mlp import *
# %%
BATCH_SIZE = 8192
FINETUNE_BATCH_SIZE = 2048_00

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
EARLYSTOP_NUM = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
with timer("Preprocessing train"):
    # train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
    train = pd.read_parquet(train_parquet)
# %%
# feat_reg_index = [0, 17, 18, 37, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58]
# feat_reg_index += list(range(60,69))
# feat_reg_index += [89, 101, 108, 113, 119, 120, 121, 122, 124, 125, 126, 128]
# feat_spike_index_temp = list(set(range(130)).difference(feat_reg_index))
# features_reg = [f'feature_{i}' for i in feat_reg_index]
# features_spike = [f'feature_{i}' for i in feat_spike_index_temp]


# %%
# feat_spike_index = [eval(s) for s in feat_spike_index]
# for f in feat_spike_index:
#     print(f'{f},', end=' ')
# %%
feat_spike_index = [1, 2, 3, 4, 5, 6, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85, 86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 112, 115, 118]
feat_reg_index = list(set(range(130)).difference(feat_spike_index))
features_reg = [f'feature_{i}' for i in feat_reg_index]
features_spike = [f'feature_{i}' for i in feat_spike_index]
resp_cols  = ['resp_1', 'resp_2', 'resp_3','resp', 'resp_4', ]    
feat_cols = [f'feature_{i}' for i in range(130)]
cat_cols = [f+'_c' for f in features_spike]

#%%

feat_spike_index = []
for feat in tqdm(features_spike):
    sorted_counts = train[feat].value_counts().sort_values(ascending=False)
    if sorted_counts.iloc[0]/sorted_counts.iloc[1] > 30 and sorted_counts.iloc[0] > 10000:
        feat_spike_index.append(sorted_counts.name.split('_')[-1])
        most_common_val = sorted_counts.index[0]
        train[feat+'_c'] = ((train[feat]-most_common_val).abs() < 1e-6).astype(int)
        # print(sorted_vals.head(5), '\n\n')

#%%
class SpikeNet(nn.Module):
    def __init__(self, hidden_size=256, 
                       cat_dim=len(cat_cols),
                       output_size=len(resp_cols), 
                       input_size=len(feat_cols),
                       dropout_rate=0.2):
        super(SpikeNet, self).__init__()
        self.embed = nn.Embedding(cat_dim, 1)
        self.emb_dropout = nn.Dropout(0.2)

        self.batch_norm0 = nn.BatchNorm1d(input_size)
        self.dropout0 = nn.Dropout(0.1)

        self.dense1 = nn.Linear(input_size+cat_dim, hidden_size)
        nn.init.kaiming_normal_(self.dense1.weight.data)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(hidden_size+input_size+cat_dim, hidden_size)
        nn.init.kaiming_normal_(self.dense2.weight.data)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(hidden_size+hidden_size, hidden_size)
        nn.init.kaiming_normal_(self.dense3.weight.data)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.dense4 = nn.Linear(hidden_size+hidden_size, output_size)
        nn.init.kaiming_normal_(self.dense4.weight.data)

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, x_cat):
        x = self.batch_norm0(x)
        x = self.dropout0(x)
        x_cat = self.embed(x_cat.long()).squeeze()
        x = torch.cat([x, x_cat], dim=1)

        x1 = self.dense1(x)
        x1 = self.batch_norm1(x1)
        x1 = self.LeakyReLU(x1)
        x1 = self.dropout1(x1)

        x = torch.cat([x, x1], 1)

        x2 = self.dense2(x)
        x2 = self.batch_norm2(x2)
        x2 = self.LeakyReLU(x2)
        x2 = self.dropout2(x2)

        x = torch.cat([x1, x2], 1)

        x3 = self.dense3(x)
        x3 = self.batch_norm3(x3)
        x3 = self.LeakyReLU(x3)
        x3 = self.dropout3(x3)

        x = torch.cat([x2, x3], 1)

        x = self.dense4(x)

        return x
# %%
train = train.query ('date > 85').reset_index (drop = True)
train.fillna(train.mean(),inplace=True)
train = train[train['weight'] != 0].reset_index (drop = True)
train['action'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')

valid = train.loc[train.date >= 475].reset_index(drop=True)
train = train.loc[train.date < 470].reset_index(drop=True)
#%%

class MarketDatasetCat:
    def __init__(self, df, features=feat_cols, 
                           cat_features=None,
                           targets=target_cols,
                           resp = resp_cols,
                           date='date',
                           weight='weight'):
        self.features = df[features].values
        self.label = df[targets].astype('int').values.reshape(-1, len(targets))
        self.resp = df[resp].astype('float').values.reshape(-1, len(resp))
        self.date = df[date].astype('int').values.reshape(-1,1)
        self.weight = df[weight].astype('float').values.reshape(-1,1)
        self.cat_features = df[cat_features].astype('int').values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
        'features': torch.tensor(self.features[idx], dtype=torch.float),
        'cat_features': torch.tensor(self.cat_features[idx], dtype=torch.int8),
        'label': torch.tensor(self.label[idx], dtype=torch.float),
        'resp':torch.tensor(self.resp[idx], dtype=torch.float),
        'date':torch.tensor(self.date[idx], dtype=torch.int32),
        'weight':torch.tensor(self.weight[idx], dtype=torch.float),
        }

train_set = MarketDatasetCat(train, 
                                  features=feat_cols, cat_features=cat_cols, 
                                  targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = MarketDatasetCat(valid, features=feat_cols, cat_features=cat_cols, 
                                  targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# %%
util_cols = resp_cols
resp_index = [resp_cols.index(r) for r in util_cols]
regularizer = UtilityLoss(alpha=5e-2, scaling=12, normalize=None, resp_index=resp_index)
loss_fn = SmoothBCEwLogits(smoothing=0.005)

model = SpikeNet()
model.to(device);
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
#                                                     steps_per_epoch=len(train_loader),
#                                                     epochs=EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, T_mult=2, 
                                                                 eta_min=LEARNING_RATE*1e-3, last_epoch=-1)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)

early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", save_threshold=5900)

# %%
_fold = 7
SEED = 802
get_seed(SEED+SEED*_fold)
lr = []

for epoch in range(EPOCHS):

    train_loss = train_epoch_cat(model, optimizer, scheduler,loss_fn, train_loader, device)
    # train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    lr.append(optimizer.param_groups[0]['lr'])
    # if (epoch+1) % 10 == 0:
    #     _ = train_epoch_finetune(model, finetune_optimizer, scheduler,
    #                              regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device, cat_input=True)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR + \
        f"/emb_fold_{_fold}_ep_{epoch}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(epoch, valid_auc, model, model_path=model_file,
               epoch_utility_score=valid_score)
    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {_fold}")
    tqdm.write(
        f"Train loss: {train_loss:.4f} \t Current learning rate: {lr[-1]:.4e}")
    tqdm.write(
        f"Best util: {early_stop.best_utility_score:.2f} at epoch {early_stop.best_epoch} \t {early_stop.message} ")
    tqdm.write(
        f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break
# %%