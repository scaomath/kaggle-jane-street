#%%
import os, sys
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
    train['action'] = train['action'] & ((train['resp_'+str(c)] > 0))

features = [c for c in train.columns if 'feature' in c]
#%%
f_mean = np.mean(train[features[1:]].values, axis=0)
train.fillna(train.mean(),inplace=True)
#%%
train['action'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')

valid = train.loc[train.date >= 450].reset_index(drop=True)
train = train.loc[train.date <= 425].reset_index(drop=True)

target_cols = ['action', 'action_1', 'action_2', 'action_3', 'action_4']

#%%
all_feat_cols = [col for col in feat_cols]

train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
valid['cross_41_42_43'] = valid['feature_41'] + valid['feature_42'] + valid['feature_43']
valid['cross_1_2'] = valid['feature_1'] / (valid['feature_2'] + 1e-5)

all_feat_cols.extend(['cross_41_42_43', 'cross_1_2'])


# %%
train_set = MarketDataset(train, features=all_feat_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
valid_set = MarketDataset(valid, features=all_feat_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
# %%
model = ResidualMLP()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer_ = Lookahead(optimizer=optimizer, k=10, alpha=0.5)
scheduler = None
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                                 max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(train_loader))
loss_fn = SmoothBCEwLogits(smoothing=0.01)
model_file = MODEL_DIR+f"/pt_resmlp_{SEED}.pth"
es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
# %%
start_time = time()
pbar = tqdm(total=EPOCHS)
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc = roc_auc_score(valid[target_cols].values.reshape(-1), valid_pred)
    valid_logloss = log_loss(valid[target_cols].values.reshape(-1), valid_pred)
    valid_pred = np.median(valid_pred, axis=1)
    valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
    valid_u_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values,
                                            resp=valid.resp.values, action=valid_pred)
    print(f"EPOCH:{epoch:3} train_loss={train_loss:.5f} "
                f"valid_u_score={valid_u_score:.5f} valid_auc={valid_auc:.5f} "
                f"time: {(time() - start_time) / 60:.2f}min")
    es(valid_auc, model, model_path=model_file)
    if es.early_stop:
        print("Early stopping")
        torch.save(model.state_dict(), model_file)
        break
    pbar.update()

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
    valid_score = utility_score_bincount(date=valid.date.values, weight=valid.weight.values, resp=valid.resp.values,
                                            action=valid_pred)
    print(f'{NFOLDS} models valid score: {valid_score}\tauc_score: {auc_score:.4f}\tlogloss_score:{logloss_score:.4f}')
# %%
