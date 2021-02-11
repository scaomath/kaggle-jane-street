#%%
import os, sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from   fastai.tabular.all import TabularPandas, RandomSplitter, CategoryBlock, MultiCategoryBlock, range_of, accuracy, tabular_learner, TabularDataLoaders

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)

from utils import *
from utils_js import *
#%%
dtype = {
    'feature'  : 'str', 
    'tag_0'    : 'int8'
}
for i in range (1, 29):
    k = 'tag_' + str (i)
    dtype[k] = 'int8'

features_df = pd.read_csv (os.path.join(DATA_DIR, 'features.csv'), usecols=range(1,30), dtype=dtype)
N_FEATURES  = features_df.shape[0]  # the features.csv has 130 features (1st row) = no of features in train.csv (feature_0 to feature_129)
N_FEAT_TAGS = features_df.shape[1]  # the features.csv has 29 tags

resp_cols  = ['resp_1', 'resp_2', 'resp_3','resp_4', 'resp']    
feat_cols = [f'feature_{i}' for i in range(130)]
# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
    train = train.query ('date > 85').reset_index (drop = True)
        # df = df[df['weight'] != 0].reset_index (drop = True)
y = np.stack ([(train[c] > 0).astype ('int') for c in resp_cols]).T
# train.drop (columns=['weight', 'date', 'ts_id']+resp_cols, inplace=True)

train.fillna(train.mean(),inplace=True)
# %%
splits  = RandomSplitter (valid_pct=0.05) (range_of (df))

to        = TabularPandas (df, cont_names=feat_cols, cat_names=None, y_names=resp_cols, 
                            y_block=MultiCategoryBlock(encoded=True, vocab=resp_cols), splits=splits)