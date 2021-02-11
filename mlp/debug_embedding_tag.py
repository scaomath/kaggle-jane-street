#%%
import os, sys
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torchsummary import summary
# from fastai.tabular.all import TabularPandas, RandomSplitter, CategoryBlock, MultiCategoryBlock, range_of, accuracy, tabular_learner, TabularDataLoaders

current_path = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(current_path)
MODEL_DIR = os.path.join(HOME,  'models')
DATA_DIR = os.path.join(HOME,  'data')
sys.path.append(HOME)

from mlp import *
from utils import *
from utils_js import *
#%%

THREE_HIDDEN_LAYERS = [400, 400, 400]
N_FEAT_TAGS = 29
N_TARGETS = 6

BATCH_SIZE = 8196
EARLYSTOP_NUM = 5
FINETUNE_BATCH_SIZE = 51200

EPOCHS = 100

N_DENOISED_TARGET = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

N_FEATURES = 130
N_FEAT_TAGS = 29

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dtype = {
    'feature'  : 'str', 
    'tag_0'    : 'int8'
}
for i in range (1, 29):
    k = 'tag_' + str (i)
    dtype[k] = 'int8'

features_df = pd.read_csv (os.path.join(DATA_DIR, 'features.csv'), usecols=range(1,30), dtype=dtype)
# N_FEATURES  = features_df.shape[0]  # the features.csv has 130 features (1st row) = no of features in train.csv (feature_0 to feature_129)
# N_FEAT_TAGS = features_df.shape[1]  # the features.csv has 29 tags

resp_cols  = ['resp_1', 'resp_2', 'resp_3','resp_4', 'resp']    
feat_cols = [f'feature_{i}' for i in range(130)]
resp_cols = ['resp', 'resp_dn_0', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
target_cols = ['action', 'action_dn_0', 'action_1', 'action_2', 'action_3', 'action_4']
# %%
with timer("Preprocessing train"):
    train_parquet = os.path.join(DATA_DIR, 'train.parquet')
    train = pd.read_parquet(train_parquet)
    train = train.query ('date > 85').reset_index (drop = True)
        # df = df[df['weight'] != 0].reset_index (drop = True)

train.fillna(train.mean(),inplace=True)
train = add_denoised_target(train, num_dn_target=N_DENOISED_TARGET)
y = np.stack ([(train[c] > 0).astype ('int') for c in resp_cols]).T
# train.drop (columns=['weight', 'date', 'ts_id']+resp_cols, inplace=True)
train['action'] = (train['resp'] > 0).astype('int')
for c in range(1,5):
    train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype('int')
valid = train.loc[train.date > 450].reset_index(drop=True)
# %%
class FeatureFFN (nn.Module):
    
    def __init__(self, inputCount=130, 
                 outputCount=5, 
                 hiddenLayerCounts=[150, 150, 150], 
                 drop_prob=0.2, 
                 activation=nn.SiLU() # this is swish activation
                 ):
        '''
        Feature generation embedding net, no output
        '''
        super(FeatureFFN, self).__init__()
        
        self.activation = activation
        self.dropout    = nn.Dropout (drop_prob)
        self.batchnorm0 = nn.BatchNorm1d (inputCount)
        self.dense1     = nn.Linear (inputCount, hiddenLayerCounts[0])
        self.batchnorm1 = nn.BatchNorm1d (hiddenLayerCounts[0])
        self.dense2     = nn.Linear(hiddenLayerCounts[0], hiddenLayerCounts[1])
        self.batchnorm2 = nn.BatchNorm1d (hiddenLayerCounts[1])
        self.dense3     = nn.Linear(hiddenLayerCounts[1], hiddenLayerCounts[2])
        self.batchnorm3 = nn.BatchNorm1d (hiddenLayerCounts[2])        
        self.outDense   = None
        if outputCount > 0:
            self.outDense   = nn.Linear(hiddenLayerCounts[-1], outputCount)

    def forward (self, x):
        
        # x = self.dropout (self.batchnorm0 (x))
        x = self.batchnorm0(x)
        x = self.dropout (self.activation (self.batchnorm1 (self.dense1 (x))))
        x = self.dropout (self.activation (self.batchnorm2 (self.dense2 (x))))
        x = self.dropout (self.activation (self.batchnorm3 (self.dense3 (x))))
        # x = self.outDense (x)
        return x
# %%
class EmbedFNN (nn.Module):
    
    def __init__(self, three_hidden_layers=THREE_HIDDEN_LAYERS, 
                       embed_dim=N_FEAT_TAGS, 
                       features_tag_matrix=features_df):
        
        super(EmbedFNN, self).__init__()
        global N_FEAT_TAGS
        N_FEAT_TAGS = 29
        
        # store the features to tags mapping as a datframe tdf, feature_i mapping is in tdf[i, :]
        # dtype = {'tag_0' : 'int8'}
        # for i in range (1, 29):
        #     k = 'tag_' + str (i)
        #     dtype[k] = 'int8'
        # t_df = pd.read_csv ('features.csv', usecols=range (1,N_FEAT_TAGS+1), dtype=dtype)
        # tag_29 is for feature_0
        features_tag_matrix['tag_29'] = np.array ([1] + ([0]*(N_FEATURES-1)) ).astype ('int8')
        self.features_tag_matrix = torch.tensor(features_tag_matrix.values, dtype=torch.float32)
        # torch.tensor(t_df.to_numpy())
        N_FEAT_TAGS += 1
        
        
        # embeddings for the tags. Each feature is taken a an embedding which is an avg. of its' tag embeddings
        self.embed_dim  = embed_dim
        self.tag_embedding = nn.Embedding(N_FEAT_TAGS+1, embed_dim) # create a special tag if not known tag for any feature
        self.tag_weights = nn.Linear(N_FEAT_TAGS, 1)
        
        drop_prob = 0.5
        self.ffn = FeatureFFN(inputCount=(N_FEATURES+embed_dim), 
                             outputCount=0, 
                             hiddenLayerCounts=[(three_hidden_layers[0]+embed_dim), 
                                               (three_hidden_layers[1]+embed_dim), 
                                               (three_hidden_layers[2]+embed_dim)], 
                             drop_prob=drop_prob)
        self.outDense = nn.Linear (three_hidden_layers[2]+embed_dim, N_TARGETS)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return
    
    def features2emb (self):
        """
        idx : int feature index 0 to N_FEATURES-1 (129)
        """
        
        all_tag_idxs = torch.LongTensor(np.arange(N_FEAT_TAGS)) # (29,)
        tag_bools = self.features_tag_matrix.to(self.device) # (130, 29)
        # print ('tag_bools.shape =', tag_bools.size())
        all_tag_idxs = all_tag_idxs.to(self.device)
        f_emb = self.tag_embedding(all_tag_idxs).repeat(N_FEATURES, 1, 1)    
        #;print ('1. f_emb =', f_emb) # (29, 7) * (130, 1, 1) = (130, 29, 7)
        # print ('f_emb.shape =', f_emb.size())
        f_emb = f_emb * tag_bools[:, :, None]                           
        #;print ('2. f_emb =', f_emb) # (130, 29, 7) * (130, 29, 1) = (130, 29, 7)
        # print ('f_emb.shape =', f_emb.size())
        
        # Take avg. of all the present tag's embeddings to get the embedding for a feature
        s = torch.sum (tag_bools, dim=1) # (130,)       
        f_emb = torch.sum (f_emb, dim=-2) / s[:, None]
        # (130, 7)
        # print ('f_emb =', f_emb)        
        # print ('f_emb.shape =', f_emb.shape)
        
        # take a linear combination of the present tag's embeddings
        # f_emb = f_emb.permute (0, 2, 1) # (130, 7, 29)
        # f_emb = self.tag_weights (f_emb)                      
        # #;print ('3. f_emb =', f_emb)    # (130, 7, 1)
        # f_emb = torch.squeeze (f_emb, dim=-1)                 
        # #;print ('4. f_emb =', f_emb)   # (130, 7)
        return f_emb.detach().to(self.device)
    
    def forward (self, features, cat_featrs=None):
        """
        when you call `model (x ,y, z, ...)` then this method is invoked
        """
        
        # cat_featrs = None
        features   = features.view (-1, N_FEATURES)
        f_emb = self.features2emb()
        features_2 = torch.matmul (features, f_emb)
        
        # Concatenate the two features (features + their embeddings)
        features = torch.hstack ((features, features_2))       
        
        x = self.ffn(features)
        out = self.outDense(x)
        return out

# %%
train_set = ExtendedMarketDataset(train, features=feat_cols, targets=target_cols, resp=resp_cols)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

valid_set = ExtendedMarketDataset(valid, features=feat_cols, targets=target_cols, resp=resp_cols)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = EmbedFNN()
model.to(device);
# %%
util_cols = resp_cols
resp_index = [resp_cols.index(r) for r in util_cols]

regularizer = UtilityLoss(alpha=5e-2, scaling=12, normalize=None, resp_index=resp_index)

loss_fn = SmoothBCEwLogits(smoothing=0.005)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer = RAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE,
#                                                     steps_per_epoch=len(train_loader),
#                                                     epochs=EPOCHS)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, T_mult=1, 
                                                                 eta_min=LEARNING_RATE*1e-3, last_epoch=-1)

finetune_loader = DataLoader(
    train_set, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=8)

finetune_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE*1e-3)

early_stop = EarlyStopping(patience=EARLYSTOP_NUM, mode="max", save_threshold=5900)

# %%
_fold = 7
SEED = 802
get_seed(SEED+SEED*_fold)
lr = []

for epoch in range(EPOCHS):

    train_loss = train_epoch(model, optimizer, scheduler,loss_fn, train_loader, device)
    # train_loss = train_epoch_weighted(model, optimizer, scheduler, loss_fn, train_loader, device)
    lr.append(optimizer.param_groups[0]['lr'])
    if (epoch+1) % 10 == 0:
        _ = train_epoch_finetune(model, finetune_optimizer, scheduler,
                                 regularizer, finetune_loader, device, loss_fn=loss_fn)

    valid_pred = valid_epoch(model, valid_loader, device)
    valid_auc, valid_score = get_valid_score(valid_pred, valid,
                                             f=median_avg, threshold=0.5, target_cols=target_cols)
    model_file = MODEL_DIR + \
        f"/dn_ep_{epoch}_util_{int(valid_score)}_auc_{valid_auc:.4f}.pth"
    early_stop(valid_auc, model, model_path=model_file,
               epoch_utility_score=valid_score)
    tqdm.write(f"\n[Epoch {epoch+1}/{EPOCHS}] \t Fold {_fold}")
    tqdm.write(
        f"Train loss: {train_loss:.4f} \t Current learning rate: {lr[-1]:.4e}")
    tqdm.write(
        f"Best util: {early_stop.best_utility_score:.2f} \t {early_stop.message} ")
    tqdm.write(
        f"Valid utility: {valid_score:.2f} \t Valid AUC: {valid_auc:.4f}\n")
    if early_stop.early_stop:
        print("\nEarly stopping")
        break