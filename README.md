# Playground for Jane Street Market Prediction Competition on Kaggle

# Team: Semper Augustus
Shuhao Cao, Carl McBride Ellis, Ethan Zheng


# TO-DO for final submission:
- [x] EDA.
- [x] A simple starter.
- [x] Stable CV-LB strategy (Updated Jan 22, now I think this is somehow impossible; updated Feb 12, certain correlation between the LB and the denoised target utility-finetuning around 70 epochs of ADAM).
- [x] Writing a simple `iter_env` simulator.
- [ ] Testing a moving average `fillna()` strategy in both train and inference pipeline.
- [ ] Testing a past mean `fillna()`, fill the NaN using the mean only from prior day data, no intraday data.
- [x] Using the `iter_env` simulator to test the impact of different threshold: 0.502 or 0.498 can be both better than 0.5? Need an explanation...
- [ ] A table compiling what features will be using `ffill`, previous day mean, overall mean, etc (maybe not necessary?).
- [ ] Trading frequency can be determined by number of trades per day, store this in a cache to choose model.
- [ ] Using `feature_0` to choose models, and/or threshold (based on `feature_0`'s previous day count).
- [ ] Using rolling mean of previous days as input, working out a submission pipeline.
- [x] Implement a regularizer using the utility function.
- [x] Train with all weights (maybe making `weight==0` rows' weights to certain small number `1e-5`), then train with all positive `weight` rows (slightly better public leaderboard).
- [x] Train with a weighted cross entropy loss, the weight is $\ln(1+w)$; the local CV became better but public leaderboard became worse. 
- [x] Adding one or multiple denoised targets by removing the eigenvalues of the covariance matrix.
- [ ] Train a model including the first 85 days (high volatility days).

# Ideas and notes
- Final sub: 1 with best public LB+CV, 1 experimental.
- submission idea: using different models for the `iter_env`, see if we can extract features representing general market trend, then switch/ensemble different models. Especially when the market is in volatile, the strategy in general needs to be more conservative.
-  `feature_64` may be used as time to build CV strategy. Ref: [https://www.kaggle.com/c/jane-street-market-prediction/discussion/202253](https://www.kaggle.com/c/jane-street-market-prediction/discussion/202253)
- Number of trades `ts_id` might be related to volatility on that day: [https://www.kaggle.com/c/jane-street-market-prediction/discussion/201930#1125847](https://www.kaggle.com/c/jane-street-market-prediction/discussion/201930#1125847)
- Another notebook on volatility (this one we should use its info at dicretion): [https://www.kaggle.com/charlesleahy/identifying-volatility-features](https://www.kaggle.com/charlesleahy/identifying-volatility-features)
- Two-Sigma's stock prediction competition (with a similar format) winning solution of 5th place: [https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd](https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd)
- The data in the actual `test` is disjoint from the `train` (confirmed by the host at [https://www.kaggle.com/c/jane-street-market-prediction/discussion/199551](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199551))



# EDA
- Only 35%-40% of the samples have `action` being 1, depending on the CV split.

# NN models
Current NN models use `date>85` and `weight>0`.
- Current best: Ethan's AE+MLP baseline the last 2 folds, not fine-tune models, with a custom median ensembling.
- (After debugging) Both custom median (average of middle 50%) and `np.mean` has better public score.
- Current NN models uses `fillna` either with mean or forward fill, mean performs better on public LB but certain is subject to leakage. 

## Autoencoder
- Kaggle Notebook: [https://www.kaggle.com/ztyreg/fork-of-s8900-ts](https://www.kaggle.com/ztyreg/fork-of-s8900-ts)
- Local Notebook: TBD
- Score: 8358.763
- Submission time: ~2 hours (CPU)

**Thoughts**:
- Forward fill (8781.740) seems to be better than mean imputation, although I haven't tested if the difference is significant

## AE+MLP+prediction cache
- Attempt 0.1: simply saving `pred_df.copy()` and using `pd.concat` is way too slow (7-8 iteration/s << 45 which is the current starter's). TO-DO: add a class so that prediction is a function under this class, model outputs to give more information, and some objects "depicting" the current market volatility. 

## A new Residual+MLP model
- The key is to train using the actual `resp` columns as target, and when doing the inference, apply the sigmoid function to the output (why `BCEwLogits` performs better than `CrossEntropy`???).
- Set up the baseline training, adding a 16-target model (using various sums between the `resp` columns).
- Tested the sensitivity of seeds to the CV vs public leaderboard. Bigger model in general is less sensitive than smaller models (esp the seed 1111 overfit model).
- A local-public LB stable training strategy: RAdam/Adam with cosine annealing scheduler, utility function regularizer finetuning every 10 epochs with a `1e-3*lr` learning rate, 1 or 2 denoised targets added, 50% median average ensembling. 
- Feature neutralization might not fit the iteration speed needed for the inference.

# Gradient boosting models
## XGBoost:
- Kaggle Notebook: [https://www.kaggle.com/ztyreg/xgb-benchmark](https://www.kaggle.com/ztyreg/xgb-benchmark)
- Local Notebook: [https://github.com/scaomath/kaggle-jane-street/blob/main/lgb/v01_explore.ipynb](https://github.com/scaomath/kaggle-jane-street/blob/main/lgb/v01_explore.ipynb)
- Score: 5557.170
- Submission time: ~2 hours (CPU)

**Notes**:
- Training 1 XGBoost model only takes about 5 minutes, so we do not need to save the model
- Needs different feature processing than the autoencoder model

**Thoughts**:
- Add time lag features
  - Add all lag1 features: no improvement (5039.022)
- Add transformed features (abs, log, std, polynomial)

# Trained models
Google drive folder: TBD

# Repo structure

```bash
├── model
│   └── model dumps: hdf5, pt, etc
├── data
│   ├── EDA ipynbs
│   ├── processed data
│   └── raw data
├── nn
├── transformer
├── lgb
├── data.py: comepetition data downloader
├── utils.py: utility functions
├── README.md: can be used as a log
└── .gitignore
```
