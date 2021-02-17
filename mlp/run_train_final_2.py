# %%
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

from utils import *
from mlp import *
# %%
'''
Training script (excluding volatile days):
1. data: after day 85, excluding (2, 294, 36, 270)
2. data: the fillna is using the past day mean (after excluding the days above)
3. data: all five resps
4. training: finetuning using resp columns as regularizer, every 5 iterations
'''

DEBUG = False
LOAD_PRETRAIN = False
TRAINING_START = 0 
FINETUNE_BATCH_SIZE = 2048_00
BATCH_SIZE = 8192
EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLYSTOP_NUM = 20
NFOLDS = 1
SCALING = 12
THRESHOLD = 0.5
CV_THRESH = 6000
DAYS_TO_DROP = list(range(86))+[270, 294]
VOLATILE_DAYS = [1, 3, 4, 5, 8, 9, 12, 16, 17, 18, 23, 24, 26, 27, 30, 31, 32, 37, 38, 
                 41, 43, 44, 45, 46, 47, 59, 63, 69, 80, 85, 161, 168, 185, 196, 223, 231, 235, 
                 262, 274, 276, 283, 324, 346, 353, 354, 356, 379, 380, 382, 393, 394, 427, 438, 
                 452, 454, 459, 462, 468, 475, 488, 489, 491, 492, 495]

SEED = 1127802
get_seed(SEED)

resp_cols = ['resp_1','resp_2', 'resp_3','resp', 'resp_4']
resp_cols_all = resp_cols
target_cols = ['action_1','action_2','action_3', 'action', 'action_4']
feat_cols = [f'feature_{i}' for i in range(130)]
feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

###### adding weight to the features #######
# feat_cols.extend(['weight'])

util_cols =['resp_1','resp_2', 'resp_3','resp', 'resp_4']
resp_index = [resp_cols_all.index(r) for r in util_cols]


f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
    train, valid = preprocess_final(train_parquet, day_start=TRAINING_START, 
                                    training_days=range(0,470), valid_days=range(475, 500),
                                    drop_days=DAYS_TO_DROP,
                                    drop_zero_weight=True, denoised_resp=False)

# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = ResidualMLP(input_size=len(feat_cols), hidden_size=256, output_size=len(target_cols))
# model = MLP(hidden_units=(None,160,160,160), input_dim=len(feat_cols), output_dim=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))
# %%
regularizer = UtilityLoss(alpha=5e-2, scaling=SCALING, normalize=None, resp_index=resp_index)

loss_fn = SmoothBCEwLogits(smoothing=0.005)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, T_mult=2, 
                                                                 eta_min=LEARNING_RATE*1e-3, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
#                                                 steps_per_epoch=len(train_loader), epochs=EPOCHS)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=10)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-4)
early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", save_threshold=CV_THRESH)
# %%
_fold = 4
SEED = 1127802
get_seed(SEED+SEED*_fold)

for epoch in range(EPOCHS):

    train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)
    # train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    # scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    if (epoch+1) % 5 == 0:
        _ = train_epoch_finetune(model, finetune_optimizer, scheduler,
                                 regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)

    model_file = MODEL_DIR + f"/final_{_fold}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(epoch, valid_auc, model, 
               model_path=model_file,
               epoch_utility_score=valid_score)
    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {_fold}")
    tqdm.write(f"Train loss: {train_loss:.4e} \t Current learning rate: {lr:.4e}")
    tqdm.write(f"Best util: {early_stop.best_utility_score:.2f} at epoch {early_stop.best_epoch} \t {early_stop.message} ")
    tqdm.write(f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break

if DEBUG:
    torch.save(model.state_dict(), MODEL_DIR + f"/model_{_fold}.pth")
# %%
if LOAD_PRETRAIN:
    _fold = 4
    model_file = f"resw_interleave_1_util_6455_auc_0.6237.pth"
    print(f"Loading {model_file} for cv check.\n")
    model_weights = os.path.join(MODEL_DIR, model_file)

    model.to(device)
    feat_cols = [f'feature_{i}' for i in range(130)]
    feat_cols.extend(['weight'])
    feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


    model = ResidualMLP(input_size=len(feat_cols), hidden_size=256, 
                        output_size=len(target_cols))
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_weights))
    except:
        model.load_state_dict(torch.load(
            model_weights, map_location=torch.device('cpu')))
    model.eval();

train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
train = preprocess_final(train_parquet, day_start=0, drop_zero_weight=False)

CV_START_DAY = 100
CV_DAYS = 25
print_all_valid_score(train, model, start_day=CV_START_DAY, num_days=CV_DAYS, 
                        batch_size =2*8192, f=median_avg, threshold=0.5, 
                        target_cols=target_cols, 
                        feat_cols=feat_cols,
                        resp_cols=resp_cols)
# %%
