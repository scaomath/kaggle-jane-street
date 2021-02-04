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
Training script finetuning using resp colums as regularizer
'''

DEBUG = False
LOAD_PRETRAIN = False
FINETUNE_BATCH_SIZE = 2048_00
BATCH_SIZE = 25600
EPOCHS = 50
FINETUNE_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 10
NFOLDS = 1
SCALING = 10
THRESHOLD = 0.5
SEED = 802
get_seed(SEED)

# f = np.median
# f = np.mean
f = median_avg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%

#%%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train, valid = preprocess_pt(train_parquet,drop_weight=True)

print(f'action based on resp mean:   ', train['action_0'].astype(int).mean())
for c in range(1,5):
    print(f'action based on resp_{c} mean: ', train['action_'+str(c)].astype(int).mean())

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
resp_cols_all = resp_cols
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
feat_cols = [f'feature_{i}' for i in range(130)]
# f_mean = np.mean(train[feat_cols[1:]].values, axis=0)
feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = ResidualMLP(hidden_size=512, output_size=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = None
loss_fn = SmoothBCEwLogits(smoothing=0.005)


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
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()
    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid, 
                                            f=median_avg, threshold=0.5, target_cols=target_cols)

    print(f"valid_utility:{valid_score:.2f} \t valid_auc:{valid_auc:.4f}")
# %%
'''
fine-tuning the trained model based on resp
max batch_size:
3 resps: 102400

current best setting: 
'''
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

# resp_cols = ['resp', 'resp_1', 'resp_2']
# resp_cols = ['resp']
resp_index = [resp_cols_all.index(r) for r in resp_cols] 

# regularizer = RespMSELoss(alpha=1e-1, scaling=1, resp_index=resp_index)
regularizer = UtilityLoss(alpha=1e-1, scaling=12, normalize=None, resp_index=resp_index)

all_train = pd.concat([train, valid], axis=0)
train_set = ExtendedMarketDataset(all_train, features=feat_cols, targets=target_cols, resp=resp_cols)

finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-2)

early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
# %%
EPOCHS = 100
_fold = 0
SEED = 1127802
get_seed(SEED)

for epoch in range(EPOCHS):
    tqdm.write(f"\nEpoch {epoch+1} for model {_fold}")
    train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)
    if (epoch+1) % 3 ==0:
        _ = train_epoch_finetune(model, finetune_optimizer, scheduler, 
                            regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid, 
                                            f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR+f"/resmlp_seed_{SEED}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(valid_auc, model, model_path=model_file, epoch_utility_score=valid_score)
    
    tqdm.write(f"Train loss:{train_loss:.4f}")
    tqdm.write(f"Early stop counter: {early_stop.counter} \t {early_stop.message} ")
    tqdm.write(f"\nValid utility:{valid_score:.2f} \t Valid AUC:{valid_auc:.4f}")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break

# %%
