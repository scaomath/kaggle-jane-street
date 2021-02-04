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
from utils_js import *
from mlp import *
# %%

'''
Training script finetuning using a utility regularizer
'''

DEBUG = False
FINETUNE = True
BATCH_SIZE = 4096
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

#%%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train, valid = preprocess_pt(train_parquet,drop_weight=True)

for c in range(1,5):
    print(f'action based on resp_{c} mean: ' ,' '*10, train['action_'+str(c)].astype(int).mean())

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

# sanity check
# item = next(iter(train_loader))
# print(item)
# %%
model = ResidualMLP(output_size=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
scheduler = None
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                                 max_lr=1e-2, epochs=EPOCHS, 
#                                                 steps_per_epoch=len(train_loader))
loss_fn = SmoothBCEwLogits(smoothing=0.005)

es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")

# %%
if DEBUG:
    regularizer = UtilityLoss(alpha=1e-4, scaling=10)
    data = next(iter(train_loader))
    optimizer.zero_grad()
    features = data['features'].to(device)
    label = data['label'].to(device)
    weight = data['weight'].view(-1).to(device)
    resp = data['resp'].to(device)
    date = data['date'].view(-1).to(device)
    model.eval()
    outputs = model(features)
    loss = loss_fn(outputs, label)
    # loss += regularizer(outputs, resp, weight=weight, date=date)
    # loss.backward()
# %%
if FINETUNE:
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
fine-tuning the trained model utility score
max batch_size:
3 resps: 409600
5 resps: 204800

current best setting: 
fold 0, batch_size = 409600, lr *= 1e-3, alpha=5e-2, 1 epoch with loss
fold 1, batch_size = 102400, lr *= 1e-3, 2 epochs
fold 2,  batch_size = 102400, lr *= 1e-2, 2 epochs
fold 3, batch_size = 409600, lr *= 1e-3, alpha=1e-1, 1 epoch without loss
fold 4, batch_size = 12800, lr *= 1e-2, alpha=1, 1 epoch without loss
to-do: using the least square loss to model w_{ij} res[ij]
'''
get_seed(1127)
# resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

# resp_cols = ['resp', 'resp_1', 'resp_2']
resp_cols = ['resp']
resp_index = [resp_cols_all.index(r) for r in resp_cols] # resp_1, resp_2

FINETUNE_BATCH_SIZE = 4096_00
regularizer = UtilityLoss(alpha=1e-1, scaling=12, normalize=None, resp_index=resp_index)
finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)
train_loader = DataLoader(train_set, batch_size=400_000, shuffle=True, num_workers=8)
finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)

#%%
FINETUNE_EPOCHS = 1
for epoch in range(FINETUNE_EPOCHS):
    tqdm.write(f"\nFine tuning epoch {epoch+1} for model {_fold}")
    # train_loss = train_epoch(model, finetune_optimizer, scheduler, 
    #                          loss_fn, train_loader, device)   
    _ = train_epoch_utility(model, finetune_optimizer, scheduler, 
                            regularizer, finetune_loader, device, loss_fn=loss_fn)                     
    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid, 
                                            f=median_avg, threshold=0.5, target_cols=target_cols)

    tqdm.write(f"\nval_utility:{valid_score:.2f}  valid_auc:{valid_auc:.4f}")
 # %%
torch.save(model.state_dict(), MODEL_DIR+f"/resmlp_finetune_fold_{_fold}.pth")
# %%
#%%
# regularizer = UtilityLoss(alpha=1e-4, scaling=12)

# finetune_loader = DataLoader(train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)
# finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)


# for epoch in range(EPOCHS):

#     start_time = time()
#     train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)
        
#     train_loss = train_epoch_utility(model, finetune_optimizer, scheduler, 
#                                          loss_fn, regularizer, finetune_loader, device)

#     valid_pred = valid_epoch(model, valid_loader, device)
#     valid_auc, valid_score = get_valid_score(valid_pred, valid, 
#                                         f=median_avg, threshold=0.5, target_cols=target_cols)
#     model_file = MODEL_DIR+f"/resmlp_seed_{SEED}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
#     es(valid_auc, model, model_path=model_file, epoch_utility_score=valid_score)

#     print(f"\nEPOCH:{epoch:2d} tr_loss:{train_loss:.2f}  "
#                 f"val_utility:{valid_score:.2f} valid_auc:{valid_auc:.4f}  "
#                 f"epoch time: {time() - start_time:.1f}sec  "
#                 f"early stop counter: {es.counter}\n")
    
#     if es.early_stop:
#         print("\nEarly stopping")
#         break
