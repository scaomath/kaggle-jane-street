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
Training script finetuning using resp colums as regularizer with an additional denoised target
'''

DEBUG = False
LOAD_PRETRAIN = False
TRAINING_START = 86  # 86 by default

FINETUNE_BATCH_SIZE = 2048_00
BATCH_SIZE = 8196
EPOCHS = 120
EARLYSTOP_NUM = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SCALING = 10
THRESHOLD = 0.5
NUM_DENOISE = 1
DAYS_TO_DROP = [2, 36, 270, 294]
SEED = 1127802
get_seed(SEED)

# f = np.median
# f = np.mean
f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


feat_cols = [f'feature_{i}' for i in range(130)]
# f_mean = np.mean(train[feat_cols[1:]].values, axis=0)
feat_cols.extend(['cross_41_42_43', 'cross_1_2'])

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
resp_cols_all = resp_cols
target_cols = ['action','action_1', 'action_2', 'action_3', 'action_4']

for c in range(NUM_DENOISE):
    resp_cols += [f'resp_dn_{c}']
    target_cols += [f'action_dn_{c}']


# %%
with timer("Preprocessing train"):
    # train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
    train, valid = preprocess_pt(train_parquet, 
                                 day_start=TRAINING_START, 
                                 drop_days=DAYS_TO_DROP,
                                 drop_zero_weight=False, 
                                 zero_weight_thresh=None,
                                 denoised_resp=True, 
                                 num_dn_target=NUM_DENOISE)

print(f'action based on resp mean:   ', train['action'].astype(int).mean())
print(f'action based on resp_dn_0 mean:', train[f'action_dn_0'].astype(int).mean())

for c in range(1, 5):
    print(f'action based on resp_{c} mean: ', train['action_'+str(c)].astype(int).mean())


# %%
train_set = ExtendedMarketDataset(
    train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


# %%
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4'] 
for i in range(NUM_DENOISE):
    resp_cols += [f'resp_dn_{i}']

# util_cols = ['resp', 'resp_1', 'resp_2']
# util_cols = ['resp', 'resp_4']
# util_cols = ['resp']
util_cols = resp_cols

resp_index = [resp_cols_all.index(r) for r in util_cols]

# regularizer = RespMSELoss(alpha=1e-1, scaling=1, resp_index=resp_index)
regularizer = UtilityLoss(alpha=5e-2, scaling=12, normalize=None, resp_index=resp_index)

loss_fn = SmoothBCEwLogits(smoothing=0.005)

# all_train = pd.concat([train, valid], axis=0)
# all_train_set = ExtendedMarketDataset(all_train, features=feat_cols, targets=target_cols, resp=resp_cols)
# train_loader = DataLoader(all_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


model = ResidualMLP(hidden_size=160, output_size=len(target_cols))
# model = MLP(hidden_units=(None,160,160,160), input_dim=len(feat_cols), output_dim=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = Lookahead(optimizer=optimizer, alpha=1e-1)
# scheduler = None

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
#                                                     steps_per_epoch=len(train_loader),
#                                                     epochs=EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, T_mult=2, 
                                                                 eta_min=LEARNING_RATE*1e-3, last_epoch=-1)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)

early_stop = EarlyStopping(patience=EARLYSTOP_NUM,
                           mode="max", save_threshold=5900)

# %%
if LOAD_PRETRAIN:
    print("Loading model for finetune.")
    _fold = 0
    model_weights = os.path.join(MODEL_DIR, f"resmlp_{_fold}.pth")
    # model_weights = os.path.join(MODEL_DIR, f"resmlp_ft_old_fold_{_fold}.pth")
    # model_weights = os.path.join(MODEL_DIR, f"resmlp_finetune_fold_{_fold}.pth")
    try:
        model.load_state_dict(torch.load(model_weights))
    except:
        model.load_state_dict(torch.load(
            model_weights, map_location=torch.device('cpu')))
    model.eval()
    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)

    print(f"valid_utility:{valid_score:.2f} \t valid_auc:{valid_auc:.4f}")
# %%
_fold = 6
SEED = 1127802
get_seed(SEED+SEED*_fold)
lr = []

for epoch in range(EPOCHS):

    # train_loss = train_epoch(model, optimizer, scheduler,loss_fn, train_loader, device)
    train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    lr.append(optimizer.param_groups[0]['lr'])
    if (epoch+1) % 10 == 0:
        _ = train_epoch_finetune(model, finetune_optimizer, scheduler,
                                 regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR + \
        f"/dn_fold_{_fold}_ep_{epoch}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
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

if DEBUG:
    torch.save(model.state_dict(), MODEL_DIR + f"/resmlp_interleave_fold_{_fold}.pth")

# %%
model_file = f"resmlp_interleave_0_util_7437_auc_0.6389.pth"
print(f"Loading {model_file} for cv check.")
model_weights = os.path.join(MODEL_DIR, model_file)

try:
    model.load_state_dict(torch.load(model_weights))
except:
    model.load_state_dict(torch.load(
        model_weights, map_location=torch.device('cpu')))
model.eval();

# %%
CV_START_DAY = 100
CV_DAYS = 25
all_train = pd.concat([train, valid], axis=0)
print_all_valid_score(all_train, model, start_day=CV_START_DAY, num_days=CV_DAYS, 
                        batch_size = 8192, f=median_avg, threshold=0.5, 
                        target_cols=target_cols, feat_cols=feat_cols,resp_cols=resp_cols)
# %%
