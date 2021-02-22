# %%
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
sys.path.append(HOME)

from utils import *
from utils_js import *

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


from mlp import *
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
#%%
'''
Final model spikenet:

1. subtract the most common values from columns with a spike in the histogram to form cat features.
'''


# %%
BATCH_SIZE = 8192
FINETUNE_BATCH_SIZE = 4096_00

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLYSTOP_NUM = 5
ALPHA = 0.6
EPSILON = 5e-2 # strength of the regularizer
VOLATILE_MODEL = True

s = 4
SEED = 1127*s
np.random.seed(SEED)
pd.core.common.random_state(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

splits = {
          'train_days': (range(0,457), range(0,424), range(0,391)),
          'valid_days': (range(467, 500), range(434, 466), range(401, 433)),
          }
fold = 2

if fold == 0:
    SAVE_THRESH = 1300
    VAL_OFFSET = 100
elif fold == 1:
    SAVE_THRESH = 1200
    VAL_OFFSET = 150
elif fold == 2:
    SAVE_THRESH = 90
    VAL_OFFSET = 100
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    EPSILON = 1e-2

VOLATILE_DAYS = [1,  4,  5,  12,  16,  18,  24,  37,  38,  43,  44,  45,  47,
             59,  63,  80,  85, 161, 168, 452, 459, 462]

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
# feat_spike_index = [1, 2, 3, 4, 5, 6, 14, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85, 86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 111, 112, 115, 117, 118]
feat_spike_index = [1, 2, 3, 4, 5, 6, 10, 14, 16, 69, 70, 71, 73, 74, 75, 76, 79, 80, 81, 82, 85,
                    86, 87, 88, 91, 92, 93, 94, 97, 98, 99, 100, 103, 104, 105, 106, 109, 111, 112, 115, 117, 118]
feat_reg_index = list(set(range(130)).difference(feat_spike_index))
features_reg = [f'feature_{i}' for i in feat_reg_index]
features_spike = [f'feature_{i}' for i in feat_spike_index]

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4', ]
target_cols = ['action_1', 'action_2', 'action_3', 'action', 'action_4']

feat_cols = [f'feature_{i}' for i in range(130)]
# feat_cols = features_reg
cat_cols = [f+'_c' for f in features_spike]
print(f"Number of features with spike: {len(cat_cols)}")
# %%

feat_spike_index = []
most_common_vals = []
most_common_vals = np.load(DATA_DIR+'spike_common_vals_42.npy').reshape(-1)

for i, feat in tqdm(enumerate(features_spike)):
    # sorted_counts = train[feat].value_counts().sort_values(ascending=False)
    # print(sorted_counts.head(5), '\n\n')
    # if sorted_counts.iloc[0]/sorted_counts.iloc[1] > 30 and sorted_counts.iloc[0] > 5000:
    # feat_spike_index.append(sorted_counts.name.split('_')[-1])
    # most_common_val = sorted_counts.index[0]
    # most_common_vals.append(most_common_val)
    train[feat+'_c'] = (train[feat] - most_common_vals[i]).astype(int)
    # print(train[feat+'_c'].astype(int).value_counts()[:5])
    
# %%
train = train.query(f'date not in {[2, 36, 270, 294]}').reset_index(drop=True)


if not VOLATILE_MODEL:
    train = train.query('date > 85').reset_index(drop=True)
# train = train.query(f'date not in {VOLATILE_DAYS}').reset_index(drop=True)
# train.fillna(train.mean(), inplace=True)
train = train[train['weight'] != 0].reset_index(drop=True)
train['action'] = (train['resp'] > 0).astype('int')

for c in range(1, 5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(np.int32)

valid = train.loc[train.date.isin(splits['valid_days'][fold])].reset_index(drop=True)
train = train.loc[train.date.isin(splits['train_days'][fold])].reset_index(drop=True)
# %%


train_set = MarketDatasetCat(train,
                             features=feat_cols, cat_features=cat_cols,
                             targets=target_cols, resp=resp_cols)
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = MarketDatasetCat(valid, features=feat_cols, cat_features=cat_cols,
                             targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# %%
util_cols = resp_cols
# util_cols = ['resp']
resp_index = [resp_cols.index(r) for r in util_cols]
regularizer = UtilityLoss(alpha=EPSILON, scaling=12,
                          normalize=None, resp_index=resp_index)
loss_fn = SmoothBCEwLogits(smoothing=0.005)

model = SpikeNet()
model.to(device)
summary(model, [(len(feat_cols),), (len(cat_cols),)])

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
#                                                 steps_per_epoch=len(
#                                                     train_loader),
#                                                 epochs=EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=50, T_mult=2,
                                                                 eta_min=LEARNING_RATE*1e-4, last_epoch=-1)

finetune_loader = DataLoader(
    train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-2)

early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", 
                          save_threshold=SAVE_THRESH, util_offset=VAL_OFFSET)

# %%

lr = []

for epoch in range(EPOCHS):

    train_loss = train_epoch_cat(
        model, optimizer, scheduler, loss_fn, train_loader, device)
    # train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    lr.append(optimizer.param_groups[0]['lr'])

    if (epoch+1) % 10 == 0:
        _ = train_epoch_ft_cat(model, finetune_optimizer, scheduler,
                               regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device, cat_input=True)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR + \
        f"/emb_fold_{fold}_ep_{epoch}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(epoch, valid_auc, model, model_path=model_file,
               epoch_utility_score=valid_score)

    # if early_stop.model_saved:
    #     for g in optimizer.param_groups:
    #         g['lr'] *= 0.1
    #     lr[-1] = optimizer.param_groups[0]['lr']
    #     tqdm.write(f"\nNew learning rate: {lr[-1]:.4e}")
        
    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {fold}")
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
