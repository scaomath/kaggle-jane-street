#%%
import os, sys
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)

from mlp import *
from utils import *
from utils_js import *

#%%
'''
Training script of the embedding model
'''


HIDDEN_LAYERS = [400, 400, 400] # hidden layer size for the embedding model 
N_FEATURES = 130
N_FEAT_TAGS = 29
N_TARGETS = 6
N_DENOISED_TARGET = 1

BATCH_SIZE = 8196

FINETUNE_BATCH_SIZE = 204_800

EPOCHS = 50
EARLYSTOP_NUM = 6

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feat_tag_file = os.path.join(DATA_DIR, 'features.csv')
feat_cols = [f'feature_{i}' for i in range(130)]
resp_cols = ['resp', 'resp_dn_0', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
target_cols = ['action', 'action_dn_0', 'action_1', 'action_2', 'action_3', 'action_4']

# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
    train = train.query ('date > 85').reset_index (drop = True)
        # df = df[df['weight'] != 0].reset_index (drop = True)

train.fillna(train.mean(),inplace=True)
train = add_denoised_target(train, num_dn_target=N_DENOISED_TARGET)

train['action'] = (train['resp'] > 0).astype('int')

print(f'action based on resp mean:     ', train['action'].astype(int).mean())
print(f'action based on resp_dn_{0} mean:', train[f'action_dn_{0}'].astype(int).mean())

for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(int)
    print(f'action based on resp_{c} mean:   ', train[f'action_{c}'].astype(int).mean())

valid = train.loc[train.date > 450].reset_index(drop=True)
# %%
# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

#%%
model = EmbedFNN(hidden_layers=HIDDEN_LAYERS, output_dim=len(target_cols))
model.to(device);
summary(model, input_size=(len(feat_cols), ))


util_cols = resp_cols
resp_index = [resp_cols.index(r) for r in util_cols]

regularizer = UtilityLoss(alpha=1e-1, scaling=12, normalize=None, resp_index=resp_index)

loss_fn = SmoothBCEwLogits(smoothing=0.005)



optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# scheduler = None
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                                                steps_per_epoch=len(train_loader),
                                                epochs=EPOCHS)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
#                                                                  T_0=10, T_mult=1, 
#                                                                  eta_min=LEARNING_RATE*1e-3, last_epoch=-1)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-2)
finetune_scheduler = None
early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", save_threshold=5900)
# %%
_fold = 7
SEED = 802
get_seed(SEED+SEED*_fold)
lr = []

for epoch in range(EPOCHS):

    train_loss = train_epoch(model, optimizer, scheduler,loss_fn, train_loader, device)
    lr.append(optimizer.param_groups[0]['lr'])
    if (epoch+1) % 10 == 0:
        _ = train_epoch_finetune(model, finetune_optimizer, finetune_scheduler,
                                 regularizer, finetune_loader, device, 
                                 loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR + \
        f"/emb_fold_{_fold}_ep_{epoch}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(valid_auc, model, model_path=model_file,
               epoch_utility_score=valid_score)
    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {_fold}")
    tqdm.write(
        f"Train loss: {train_loss:.4f} \t Current learning rate: {lr[-1]:.4e}")
    tqdm.write(
        f"Best util: {early_stop.best_utility_score:.2f} \t {early_stop.message} ")
    tqdm.write(
        f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break
# %%
CV_START_DAY = 100
CV_DAYS = 50
print_all_valid_score(train, model, start_day=CV_START_DAY, num_days=CV_DAYS, 
                        batch_size = 8192, f=median_avg, threshold=0.5, 
                        target_cols=target_cols, feat_cols=feat_cols, resp_cols=resp_cols)
# %%
