#%%
import os, sys
from torchsummary import summary


HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from mlp.mlp import *
# %%
BATCH_SIZE = 8192
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLYSTOP_NUM = 3
SEED = 1127
SCALING = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)

train = train.loc[train.date > 85].reset_index(drop=True)
weight_mean = train.loc[train.weight > 0].mean()
#%%
train['action'] = (train['resp'] > 0)
for c in range(1,5):
    train['action'] = train['action'] & (train['resp_'+str(c)] > 0)
print('All actions mean:       ', train['action'].astype(int).mean())

for c in range(1,5):
    train['action_0'+str(c)] = (train['resp'] > 0) & (train['resp_'+str(c)] > 0)
    print(f'action based on resp and resp_{c} mean:   ', train['action_0'+str(c)].astype(int).mean())

for i in range(1,5):
    for j in range(i+1,5):
        train['action_'+str(i)+str(j)] = (train['resp_'+str(i)] > 0)  & (train['resp_'+str(j)] > 0) 
        print(f'action based on resp_{i} and resp_{j} mean: ', train['action_'+str(i)+str(j)].astype(int).mean())

features = [c for c in train.columns if 'feature' in c]
#%%
f_mean = np.mean(train[features[1:]].values, axis=0)
train.fillna(train.mean(),inplace=True)
#%%
train['action_0'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')

valid = train.loc[train.date >= 450].reset_index(drop=True)
train = train.loc[train.date <= 425].reset_index(drop=True)

resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
weight_resp_cols = ['resp_w', 'resp_w_1', 'resp_w_2', 'resp_w_3', 'resp_w_4']
target_cols = ['action', 
               'action_01', 'action_02', 'action_03', 'action_04', 
               'action_12', 'action_13', 'action_14', 'action_23', 'action_24', 'action_34']

target_cols_ex = target_cols + resp_cols + weight_resp_cols

#%%
feat_cols = [f'feature_{i}' for i in range(130)]

train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


# %%
train_set = MarketDataset(train, features=feat_cols, targets=target_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = MarketDataset(valid, features=feat_cols,targets=target_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# %%
model = ResidualMLP(output_size=len(target_cols))
model.to(device)
summary(model, input_size=(len(feat_cols), ))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer_ = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
scheduler = None
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                                 max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(train_loader))
loss_fn = SmoothBCEwLogits(smoothing=0.01)
model_file = MODEL_DIR+f"/pt_resmlp_{SEED}.pth"
es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
# %%
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(EPOCHS):

        start_time = time()
        train_loss = train_epoch(model, optimizer_, scheduler, loss_fn, train_loader, device)

        valid_pred = valid_epoch(model, valid_loader, device)
        valid_auc = roc_auc_score(valid[target_cols].values.reshape(-1), valid_pred)
        valid_logloss = log_loss(valid[target_cols].values.reshape(-1), valid_pred)
        valid_pred = np.median(valid_pred.reshape(-1, len(target_cols)), axis=1)
        valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
        valid_u_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values,
                                                resp=valid.resp.values, action=valid_pred)
        es(valid_auc, model, model_path=model_file)

        pbar.set_description(f"EPOCH:{epoch:3} tr_loss={train_loss:.5f} "
                    f"val_u_score={valid_u_score:.5f} valid_auc={valid_auc:.5f} "
                    f"epoch time: {time() - start_time:.1f}sec "
                    f"early stop counter: {es.counter}")
        
        if es.early_stop:
            print("Early stopping")
            torch.save(model.state_dict(), model_file)
            break
        pbar.update()
#%%
if True:
    valid_pred = np.zeros((len(valid), len(target_cols)))
    for _fold in range(NFOLDS):
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = ResidualMLP()
        model.to(device)
        model_file = MODEL_DIR+f"/pt_resmlp_{SEED}.pth"
        model.load_state_dict(torch.load(model_file))

        valid_pred += valid_epoch(model, valid_loader, device) / NFOLDS
    auc_score = roc_auc_score(valid[target_cols].values, valid_pred)
    logloss_score = log_loss(valid[target_cols].values, valid_pred)

    valid_pred = np.median(valid_pred, axis=1)
    valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
    valid_score = utility_score_bincount(date=valid.date.values, 
                                         weight=valid.weight.values, 
                                         resp=valid.resp.values,
                                         action=valid_pred)
    print(f'{NFOLDS} models valid score: {valid_score}\tauc_score: {auc_score:.4f}\tlogloss_score:{logloss_score:.4f}')
# %%
