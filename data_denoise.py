#%% denoising target
import os
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, TransformerMixin

HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *
from utils_js import *
# %%

'''
By Lucas Morin
https://www.kaggle.com/lucasmorin/target-engineering-patterns-denoising
'''

def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin, eMax = var*(1-(1./q)**.5)**2, var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin,eMax,pts)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf = pd.Series(pdf.reshape(-1,), index=eVal.reshape(-1,))
    return pdf


def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec = np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:
        obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1,)
    if len(x.shape)==1:
        x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1 # numerical error
    return corr

def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error
    pdf0=mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    # Find max random eVal by fitting Marcenko’s dist
    out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),
    bounds=((1E-5,1-1E-5),))
    if out['success']:
        var=out['x'][0]
    else:
        var=1
    eMax=var*(1+(1./q)**.5)**2
    return eMax,var

def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0] - nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1

def denoisedCorr2(eVal,eVec,nFacts,alpha=0):
    # Remove noise from corr through targeted shrinkage
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts]
    eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0=np.dot(eVecL,eValL).dot(eVecL.T)
    corr1=np.dot(eVecR,eValR).dot(eVecR.T)
    corr2=corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1))
    return corr2


class RMTDenoising(BaseEstimator, TransformerMixin):
    
    def __init__(self, bWidth=.01, alpha=.5, feature_0=True, sample=0.3, seed=2021):
        self.bWidth = bWidth
        self.alpha = alpha
        self.feature_0 = feature_0
        self.sample = sample
        self.seed = seed
    
    def denoise(self, X):
        sample = X.sample(frac=self.sample, random_state=self.seed)
        q = X.shape[0] / X.shape[1]
        cov = sample.cov().values
        corr0 = cov2corr(cov)

        eVal0, eVec0 = getPCA(corr0)
        eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=self.bWidth)
        nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
        corr1 = denoisedCorr2(eVal0,eVec0,nFacts0,alpha=self.alpha)
        eVal1, eVec1 = getPCA(corr1)
        #result = np.hstack((np.diag(eVal1), var0))
        #name = [f'eigen_{i+1}' for i in range(len(eVal1))] + ['var_explained']
        return eVec1[:, :nFacts0]
    
    def fit(self, X, y=None):
        if self.feature_0:
            self.cols_ = [c for c in X.columns if c != 'feature_0']
        else:
            self.cols_ = list(X.columns)
        X_ = X[self.cols_]
        self.W_ = self.denoise(X_)
        self.dim_W_ = self.W_.shape[1]
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        names = [f'proj_{i}' for i in range(self.dim_W_)]
        projection = pd.DataFrame(fast_fillna(X_[self.cols_].values, 0).dot(self.W_), columns=names)
        if self.feature_0:
            projection['feature_0'] = X['feature_0']
        return projection
# %%
if __name__ == '__main__':
    with timer("Preprocessing train"):
        train_file = os.path.join(DATA_DIR, 'train.parquet')
        train = pd.read_parquet(train_file)

    train = train.loc[train.date > 85].reset_index(drop=True)

    #%%
    targets = ['resp','resp_1','resp_2','resp_3','resp_4']
    targets_f0 = targets + ['feature_0']
    target_tf = RMTDenoising(sample=0.8, seed=1127)

    target_tf.fit(train[targets_f0])

    targets_denoised = target_tf.transform(train[targets_f0])
    targets_denoised = targets_denoised.rename(columns={'proj_0': 'resp_dn'})
    targets_denoised[['resp_dn']] = -targets_denoised['resp_dn'].values
    print(targets_denoised.head(10))
    print(train[targets_f0].head(10))
    targets_denoised[['resp_dn']].to_csv(os.path.join(DATA_DIR,'target_dn.csv'), index=False)
