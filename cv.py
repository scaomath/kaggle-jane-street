#%%
import os, sys
import pandas as pd
import numpy as np
import datatable as dt

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 
from utils import *
from mlp.mlp import *
# %%
'''
Current CV uses train.query('date>450')
Model: pt models
'''
target_cols = ['action_0', 'action_1', 'action_2', 'action_3', 'action_4']
N_FOLDS = 5
N_MODELS = 5
BATCH_SIZE = 8192
VALID_DATE = 450
model_list = [MODEL_DIR+f'/resmlp_{i}.pth' for i in range(N_FOLDS)] # baseline

feat_cols = [f'feature_{i}' for i in range(130)]
feat_cols.extend(['cross_41_42_43', 'cross_1_2'])
# f = median_avg
f = np.median

#%%

def get_valid_df(date, fillna = 'mean'):
    data_file = find_files('train.csv', DATA_DIR)
    train = dt.fread(data_file[0]).to_pandas()
    _feat_cols = [f'feature_{i}' for i in range(130)]
    if fillna == 'mean':
        f_mean = np.mean(train[_feat_cols[1:]].values, axis=0) # for inference
        train.fillna(train.mean(),inplace=True)
    elif fillna == 'ffill':
        train[_feat_cols[1:]] = train[_feat_cols[1:]].fillna(method = 'ffill').fillna(0)
    else: # TO_DO: customized fillna_func
        pass

    train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
    train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)
    train['action_0'] = (train['resp'] > 0).astype(int)
    for c in range(1,5):
        train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(int)
        print(f'action based on resp_{c} mean:   ', train['action_'+str(c)].mean())
    valid = train.query(f'date > {date}').reset_index(drop = True) 
    valid.to_parquet(os.path.join(DATA_DIR,'valid.parquet'))

def load_models(pt_model_files):
    '''
    baseline mlp models in the mlp.mlp submodule
    '''
    assert len(pt_model_files) == NFOLDS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for model_file in pt_model_files:
        model = ResidualMLP(output_size=len(target_cols))
        model.to(device)
        try:
            model.load_state_dict(torch.load(model_file))
        except:
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        models.append(model)
    return models


def cv_score(valid_df, models, f=np.mean, thresh=0.5, device=None):
    print(f"Using {f.__qualname__} as ensembler.")
    if device is None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_pred = np.zeros((len(valid_df), len(target_cols)))
    valid_set = MarketDataset(valid_df, features=feat_cols, targets=target_cols)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    for _fold in range(len(models)):
        torch.cuda.empty_cache()
        model = models[_fold]
        valid_pred_fold = valid_epoch(model, valid_loader, device).reshape(-1, len(target_cols))
        valid_pred += valid_pred_fold / len(models)
    valid_auc = roc_auc_score(valid[target_cols].values.astype(float), valid_pred)
    logloss_score = log_loss(valid[target_cols].values.astype(float), valid_pred)

    # valid_pred = f(valid_pred[...,:len(target_cols)], axis=-1) # only first 5
    valid_pred = f(valid_pred, axis=-1) # all
    valid_pred = np.where(valid_pred >= thresh, 1, 0).astype(int)
    valid_score = utility_score_bincount(date=valid.date.values, 
                                         weight=valid.weight.values, 
                                         resp=valid.resp.values,
                                         action=valid_pred)
    valid_score_max = utility_score_bincount(date=valid.date.values, 
                                         weight=valid.weight.values, 
                                         resp=valid.resp.values,
                                         action=(valid.resp.values>0))
    print(f'Max utils score: {valid_score_max:.2f}') 
    print(f'{len(models)} models valid score: {valid_score:.2f} \t auc: {valid_auc:.4f}') 


# %%
if __name__ == '__main__':

    print(f"Current valid set is date after {VALID_DATE}.\n")
    valid_parquet = find_files('valid.parquet', DATA_DIR)
    if not valid_parquet:
        with timer("Generating validation df"):
            get_valid_df(VALID_DATE)
    else:
        with timer("Generating valid loader"):
            valid = pd.read_parquet(valid_parquet[0])
            valid_set = MarketDataset(valid, features=feat_cols, targets=target_cols)
            valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        models = load_models(model_list)
        cv_score(valid, models, f=f)


    '''
    Lindada's model scores on date > 450:
    model 0:  4948
    model 1:  5641
    model 2:  5282
    model 3:  5825
    model 4:  5849
    all five: 6165
    '''

# %%
