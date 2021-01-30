#%%
import os, sys
from numpy.lib.function_base import median
from torchsummary import summary

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from mlp import *
# %%
BATCH_SIZE = 4096
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 5
NFOLDS = 1
SCALING = 1000
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

train = train.loc[train.date > 85].reset_index(drop=True)
weight_mean = train.loc[train.weight > 0].mean()
#%%
# vanilla actions based on resp
train['action_0'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')
    print(f'action based on resp_{c} mean: ' ,' '*10, train['action_'+str(c)].astype(int).mean())

# sum
train['resp_all'] = train['resp'].copy()
for c in range(1,5):
    train['resp_all'] += train['resp_'+str(c)]
train['action'] = (train['resp_all'] > 0).astype('int')
print('All actions mean:  ', '  '*10, train['action'].astype(int).mean())

for c in range(1,5):
    train['action_0'+str(c)] = (train['resp'] + train['resp_'+str(c)] > 0)
    print(f'action based on resp and resp_{c} mean:   ', train['action_0'+str(c)].astype(int).mean())

for i in range(1,5):
    for j in range(i+1,5):
        train['action_'+str(i)+str(j)] = (train['resp_'+str(i)] + train['resp_'+str(j)] > 0) 
        print(f'action based on resp_{i} and resp_{j} mean: ', train['action_'+str(i)+str(j)].astype(int).mean())

#%%
feat_cols = [f'feature_{i}' for i in range(130)]
# feat_cols = [c for c in train.columns if 'feature' in c]
f_mean = np.mean(train[feat_cols[1:]].values, axis=0)
train.fillna(train.mean(),inplace=True)

valid = train.loc[train.date >= 450].reset_index(drop=True)
train = train.loc[train.date <= 425].reset_index(drop=True)
#%%
resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
weight_resp_cols = ['resp_w', 'resp_w_1', 'resp_w_2', 'resp_w_3', 'resp_w_4']
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
# target_cols_all = target_cols
target_cols_all = ['action', 
               'action_0', 'action_1', 'action_2', 'action_3', 'action_4', 
               'action_01', 'action_02', 'action_03', 'action_04', 
               'action_12', 'action_13', 'action_14', 'action_23', 'action_24', 'action_34']

target_cols_ex = target_cols + resp_cols + weight_resp_cols

train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


# %%
train_set = MarketDataset(train, features=feat_cols, targets=target_cols_all)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = MarketDataset(valid, features=feat_cols, targets=target_cols_all)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# %%
model = ResidualMLP(output_size=len(target_cols_all))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
scheduler = None
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                                 max_lr=1e-2, epochs=EPOCHS, 
#                                                 steps_per_epoch=len(train_loader))
loss_fn = SmoothBCEwLogits(smoothing=0.01)

es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")

# %%

with tqdm(total=EPOCHS) as pbar:
    for epoch in range(EPOCHS):

        start_time = time()
        train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)

        valid_pred = valid_epoch(model, valid_loader, device)
        valid_auc = roc_auc_score(valid[target_cols_all].values.astype(float).reshape(-1), valid_pred)
        valid_logloss = log_loss(valid[target_cols_all].values.astype(float).reshape(-1), valid_pred)
        valid_pred = valid_pred.reshape(-1, len(target_cols_all))
        # valid_pred = f(valid_pred[...,:len(target_cols)], axis=-1) # only do first 5
        valid_pred = f(valid_pred, axis=-1) # all
        valid_pred = np.where(valid_pred >= THRESHOLD, 1, 0).astype(int)
        valid_score = utility_score_bincount(date=valid.date.values, 
                                            weight=valid.weight.values,
                                            resp=valid.resp.values, 
                                            action=valid_pred)
        model_file = MODEL_DIR+f"/resmlp_seed_{SEED}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
        es(valid_auc, model, model_path=model_file, epoch_utility_score=valid_score)

        pbar.set_description(f"EPOCH:{epoch:2d} tr_loss:{train_loss:.2f}  "
                    f"val_utitlity:{valid_score:.2f} valid_auc:{valid_auc:.4f}  "
                    f"epoch time: {time() - start_time:.1f}sec  "
                    f"early stop counter: {es.counter}")
        
        if es.early_stop:
            print("\nEarly stopping")
            break
        pbar.update()
#%%
if True:
    valid_pred = np.zeros((len(valid), len(target_cols_all)))
    for _fold in range(NFOLDS):
        torch.cuda.empty_cache()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResidualMLP(output_size=len(target_cols_all))
        model.to(device)
        model_file = MODEL_DIR + '/resmlp_seed_802_util_2413_auc_0.5475.pth'
        # model_file = MODEL_DIR+f"/resmlp_seed_{SEED}_util_2217_auc_0.5526.pth"
        # model_file = MODEL_DIR + '/resmlp_seed_802_util_2229.pth'
        model.load_state_dict(torch.load(model_file))
        valid_pred_fold = valid_epoch(model, valid_loader, device).reshape(-1, len(target_cols_all))
        valid_pred += valid_pred_fold / NFOLDS
    valid_auc = roc_auc_score(valid[target_cols_all].values.astype(float), valid_pred)
    logloss_score = log_loss(valid[target_cols_all].values.astype(float), valid_pred)

    # valid_pred = f(valid_pred[...,:len(target_cols)], axis=-1) # only first 5
    valid_pred = f(valid_pred, axis=-1) # all
    valid_pred = np.where(valid_pred >= THRESHOLD, 1, 0).astype(int)
    valid_score = utility_score_bincount(date=valid.date.values, 
                                         weight=valid.weight.values, 
                                         resp=valid.resp.values,
                                         action=valid_pred)
    valid_score_max = utility_score_bincount(date=valid.date.values, 
                                         weight=valid.weight.values, 
                                         resp=valid.resp.values,
                                         action=(valid.resp.values>0))
    print(f'{NFOLDS} models valid score: {valid_score:.2f}') 
    print(f'Max possible valid score: {valid_score_max:.2f}')
    print(f'auc_score: {valid_auc:.4f} \t logloss_score: {logloss_score:.4f}')
# %%
