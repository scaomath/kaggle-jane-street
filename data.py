import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os, sys
from utils import *
import zipfile

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HOME,  'model')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME) 

'''
The API token from https://www.kaggle.com/<username>/account needs to put in ~/.kaggle/ folder in MacOS/Linux or to C:\\Users\\.kaggle\\ on Windows
'''

if __name__ == "__main__":
    print(f"Current directory     : {HOME}")
    print(f"Current data directory: {DATA_DIR}")
    data_file = find_files('train.csv', DATA_DIR)
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
    else:
        print(f"Train data at: {data_file[0]}.")