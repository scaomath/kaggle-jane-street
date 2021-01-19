#%%
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os, sys
from utils import *
import zipfile
import pandas as pd
import datatable as dt

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 

'''
The API token from https://www.kaggle.com/<username>/account needs to put in ~/.kaggle/ folder in MacOS/Linux or to C:\\Users\\.kaggle\\ on Windows
'''

#%%
if __name__ == "__main__":
    print(f"Current directory     : {HOME}")
    print(f"Current data directory: {DATA_DIR}")
    data_file = find_files('train.csv', DATA_DIR)
    data_parquet = find_files('train.parquet', DATA_DIR)
    if not data_file:
        try:
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files('jane-street-market-prediction',
                                            path=DATA_DIR, quiet=False)
            data_file = find_files('zip', DATA_DIR)
            with zipfile.ZipFile(data_file,"r") as f:
                f.extractall(DATA_DIR)
        except RuntimeError as err:
            print(f"Needs API token: {err}")
    elif data_parquet:
        print(f"Train parquet at: {data_parquet[0]}.")
        with timer("Loading train"):
            train = pd.read_parquet(data_parquet[0])
        print(train.dtypes[:10])
        print(train.dtypes[-10:])
    else:
        with timer("Processing train"):
            # train = pd.read_csv(data_file[0])
            # train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns}) 
            train_dtypes = {'date': dt.int32,
                            'ts_id': dt.int64,
                            'resp': dt.float64,
                            'weight': dt.float64,
                            'feature_0': dt.int8}
            for c in range(1,5):
                train_dtypes['resp_'+str(c)] = dt.float64
            for c in range(1,130):
                train_dtypes['feature_'+str(c)] = dt.float32

            train = dt.fread(data_file[0], 
                       columns=set(train_dtypes.keys())).to_pandas().astype(train_dtypes)
            train.to_parquet(os.path.join(DATA_DIR,'train.parquet'))