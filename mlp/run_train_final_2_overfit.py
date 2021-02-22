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
4. training: finetuning using resp columns as regularizer, every 10 iterations
'''

DEBUG = False
LOAD_PRETRAIN = False

DROP_ZERO_WEIGHT = True

TRAINING_START = 0 
FINETUNE_BATCH_SIZE = 4096_00
BATCH_SIZE = 8192
EPOCHS = 60
FINETUNE_EPOCHS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 20
NFOLDS = 1
SCALING = 12
THRESHOLD = 0.5

DAYS_TO_DROP = list(range(86))+[270, 294]
VOLATILE_DAYS = [1,  4,  5,  12,  16,  18,  24,  37,  38,  43,  44,  45,  47,
             59,  63,  80,  85, 161, 168, 452, 459, 462]
VOLATILE_MODEL = False

s = 4
SEED = 1127802*s
np.random.seed(SEED)
pd.core.common.random_state(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

splits = {
          'train_days': (range(0,500), range(0,466), range(0,433)),
          'valid_days': (range(467, 500), range(434, 466), range(401, 433)),
          }

fold = 1

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

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3','resp_4']
resp_cols_all = resp_cols
target_cols = ['action', 'action_1','action_2','action_3', 'action_4']
feat_cols = [f'feature_{i}' for i in range(130)]
feat_cols += ['cross_41_42_43', 'cross_1_2']


noisy_index = [3, 4, 5, 6, 8, 10, 12, 14, 16, 37, 38, 39, 40, 72, 73, 74, 75, 76,
                78, 79, 80, 81, 82, 83]
negative_index = [73, 75, 76, 77, 79, 81, 82]
hybrid_index = [55, 56, 57, 58, 59]
running_indices = sorted([0]+noisy_index+negative_index+hybrid_index)

rm_500_cols = ['feature_' + str(i) + '_rm_500' for i in running_indices]

#### adding the running mean
# feat_cols += rm_500_cols

###### adding weight to the features #######
# feat_cols.extend(['weight'])

util_cols =['resp_1','resp_2', 'resp_3', 'resp', 'resp_4']
# util_cols =['resp_3','resp', 'resp_4']
resp_index = [resp_cols_all.index(r) for r in util_cols]


f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train_pdm.parquet')
    train = pd.read_parquet(train_parquet)

    # feat_add_parquet = os.path.join(DATA_DIR, 'feat_rm_500.parquet')
    # feat_add_df = pd.read_parquet(feat_add_parquet)

    # train = pd.concat([train, feat_add_df], axis=1)

    if not VOLATILE_MODEL:
    # train = train.query(f'date not in {VOLATILE_DAYS}').reset_index(drop = True)
        train = train.query('date > 85').reset_index(drop=True)

    if DROP_ZERO_WEIGHT:
        train = train[train['weight'] > 0].reset_index(drop = True)
    else:
        index_zero_weight =  (train['weight']==0)
        index_zero_weight = np.where(index_zero_weight)[0]
        index_zero_weight = np.random.choice(index_zero_weight, size=int(0.4*len(index_zero_weight)))
        train.loc[index_zero_weight, ['weight']] = train.loc[index_zero_weight, ['weight']].clip(1e-7)
        # train = train[train['weight'] > 0].reset_index(drop = True)

    train['action'] = (train['resp'] > 0).astype(int)
    for c in range(1,5):
        train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(int)
    
    train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
    train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5).astype(np.float32)

    #### concat with moving mean features

    valid = train.loc[train.date.isin(splits['valid_days'][fold])].reset_index(drop=True)
    train = train.loc[train.date.isin(splits['train_days'][fold])].reset_index(drop=True)

# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = ResidualMLP(input_size=len(feat_cols), hidden_size=256, output_size=len(target_cols))
# model = MLP(hidden_units=(None,160,160,160), input_dim=len(feat_cols), output_dim=len(target_cols))
# model = ResidualMLPLite(input_size=len(feat_cols), hidden_size=256, output_size=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))
# %%
regularizer = UtilityLoss(alpha=5e-2, scaling=SCALING, normalize=None, resp_index=resp_index)

loss_fn = SmoothBCEwLogits(smoothing=0.005)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=50, T_mult=2, 
                                                                 eta_min=LEARNING_RATE*1e-3, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-8)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
#                                                 steps_per_epoch=len(train_loader), epochs=EPOCHS)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LEARNING_RATE*1e-2, 
#                                              max_lr=LEARNING_RATE, step_size_up=5, 
#                                              mode="triangular2")
# scheduler_add = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,39], gamma=0.1)
# scheduler_add = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=10)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)
early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", save_threshold=SAVE_THRESH, util_offset=VAL_OFFSET)
# %%
for epoch in range(EPOCHS):

    # train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)
    train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    # scheduler_add.step()
    lr = optimizer.param_groups[0]['lr']
    if (epoch+1) % 10 == 0:
        _ = train_epoch_finetune(model, finetune_optimizer, scheduler,
                                 regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)

    model_file = MODEL_DIR + f"/final_{fold}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(epoch, valid_auc, model, 
               model_path=model_file,
               epoch_utility_score=valid_score)

    # if early_stop.model_saved:
    #     for g in optimizer.param_groups:
    #         g['lr'] *= 0.1
    #     lr = optimizer.param_groups[0]['lr']
    #     print(f"\nNew learning rate: {lr:.4e}")

    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {fold}")
    tqdm.write(f"Train loss: {train_loss:.4e} \t Current learning rate: {lr:.4e}")
    tqdm.write(f"Best util: {early_stop.best_utility_score:.2f} at epoch {early_stop.best_epoch} \t {early_stop.message} ")
    tqdm.write(f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break

#%%
# for epoch in range(FINETUNE_EPOCHS):
#     util_loss, train_loss = train_epoch_finetune(model, finetune_optimizer, scheduler,
#                                  regularizer, finetune_loader, device, loss_fn=loss_fn)

#     valid_pred = valid_epoch(model, valid_loader, device)
#     valid_auc, valid_score = get_valid_score(valid_pred, valid,
#                                              f=median_avg, threshold=0.5, target_cols=target_cols)

#     print(f"\n[Finetune epoch {epoch+1}/{FINETUNE_EPOCHS}] \t Fold {_fold}")
#     print(f"Train loss: {train_loss:.4e} \t Util loss: {util_loss:.2f}")
#     print(f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")

# if DEBUG:
#     torch.save(model.state_dict(), MODEL_DIR + f"/model_{_fold}.pth")
# %%
print(f"Loading {early_stop.model_path} for cv check.\n")
model_weights = early_stop.model_path

feat_cols = [f'feature_{i}' for i in range(130)]
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
train = preprocess_final(train_parquet, drop_zero_weight=True)

CV_START_DAY = 0
CV_DAYS = 50
print_all_valid_score(train, model, start_day=CV_START_DAY, num_days=CV_DAYS, 
                        batch_size =2*8192, f=median_avg, threshold=0.5, 
                        target_cols=target_cols, 
                        feat_cols=feat_cols,
                        resp_cols=resp_cols)
# %%
