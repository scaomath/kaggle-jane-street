#%%
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from numba import njit
import random
import datetime

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *
from nn.mlp import *
get_system()
# %%
DEBUG = False
SEED = 1127
START_SIMU_TEST = 490 # this day to 499 as simulated test days
END_SIMU_TEST = 499
TQDM_INT = 20
batch_size = 4096
N_FOLDS = 5
N_MODELS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
with timer("Loading train parquet"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)

train['action'] = (train['resp'] > 0)
for c in range(1,5):
    train['action'] = train['action'] & ((train['resp_'+str(c)] > 0))
features = [c for c in train.columns if 'feature' in c]

f_mean = np.mean(train[features[1:]].values, axis=0)

simu_test = train.query(f'date > {START_SIMU_TEST} & date <= {END_SIMU_TEST}').reset_index(drop = True) 
print(f"Simulated public test file length: {len(simu_test)}")


# %%
class Iter_Valid(object):

    global predicted
    predicted = []

    def __init__(self, df, features, batch_size = 1):
        df = df.reset_index(drop=True)
        self.columns = ['weight'] + features + ['date']
        self.df = df[self.columns]
        self.weight = df['weight'].astype(float).values
        self.action = df['action'].astype(int).values
        self.pred_df = df[['action']]
        # self.pred_df[['action']] = 0
        self.len = len(df)
        self.current = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        pre_start = self.current
        self.current += self.batch_size
        if self.current <= self.len:
            df = self.df[pre_start:self.current].copy()
            pred_df = self.pred_df[pre_start:self.current].copy()
            return df, pred_df
        elif self.current > self.len and (self.current - self.len < self.batch_size):
            df = self.df[pre_start:self.len].copy()
            pred_df = self.pred_df[pre_start::self.len].copy()
            return df, pred_df
        else:
            raise StopIteration()

    def predict(self,pred_df):
        predicted.append(pred_df)
# %%
model_list = []
for _fold in range(N_FOLDS):
    torch.cuda.empty_cache()
    model = ResidualMLP()
    model.to(device)
    model_weights = os.path.join(MODEL_DIR, f"model_{_fold}.pth")
    try:
        model.load_state_dict(torch.load(model_weights))
    except:
        model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()
    n_params = get_num_params(model)
    print(f"Fold {_fold} model has {n_params} params.")
    model_list.append(model)

model_list = model_list[-N_MODELS:]

# %%
if __name__ == '__main__':
    '''
    inference simulation
    Using a customized class


    For the pytorch res+mlp model for day 490-499:

    5 models, np.median: 1082.92
    5 models, np.mean: 1030.73
    5 models, median avg: 1067.43
    3 models, np.median, 0.498 thresh: 1096.30
    3 models, np.median, 0.497 thresh: 1116.35
    3 models, np.median, 0.496 thresh: 1104.17
    3 models, np.mean,  0.497 thresh: 1082
    3 models, np.median, 0.502 thresh: 1088.58
    '''
    date = simu_test['date'].values
    weight = simu_test['weight'].values
    resp = simu_test['resp'].values
    action = simu_test['action'].values

    # f = np.mean # 
    # f = np.median 
    f = median_avg 

    thresh = 0.502
    print(f"\n\nPredicting the action using {thresh:.3f} threshold with {N_MODELS} models.")
    iter_test = Iter_Valid(simu_test, features)
    start = time()

    pbar = tqdm(total=len(simu_test))
    for idx, (test_df, pred_df) in enumerate(iter_test):

        if test_df['weight'].item() > 0:
            x_tt = test_df.loc[:, features].values
            if np.isnan(x_tt[:, 1:].sum()):
                x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean

            cross_41_42_43 = x_tt[:, 41] + x_tt[:, 42] + x_tt[:, 43]
            cross_1_2 = x_tt[:, 1] / (x_tt[:, 2] + 1e-5)
            feature_inp = np.concatenate((x_tt,
                                          np.array(cross_41_42_43).reshape(x_tt.shape[0], 1),
                                          np.array(cross_1_2).reshape(x_tt.shape[0], 1)), axis=1)
            pred = np.zeros((1, len(target_cols)))
            for model in model_list:
                pred += model(torch.tensor(feature_inp, dtype=torch.float).to(device))\
                                        .sigmoid().detach().cpu().numpy() / N_MODELS
            pred = f(pred.squeeze())
            pred_df.action = np.where(pred >= thresh, 1, 0).astype(int)
        else:
            pred_df.action = 0

        iter_test.predict(pred_df)

        time_taken = time() - start
        total_time_est = time_taken / (idx+1) * 1000000 / 60
        pbar.set_description(f"Current speed = {total_time_est:.1f} minutes to complete inference")
        pbar.update()

    y_true = simu_test['action']
    y_pred = pd.concat(predicted)['action']
    print('\nValidation auc:', roc_auc_score(y_true, y_pred))
    score = utility_score_bincount(date, weight, resp, y_true)
    score_pred = utility_score_bincount(date, weight, resp, y_pred)
    print('\nMax possible utility score:', score)
    print('\nModel utility score:       ', score_pred)