# Playground for Jane Street Market Prediction Competition on Kaggle


# Ideas and notes
- Final sub: 1 with best public LB+CV, 1 experimental.
- submission idea: using different models for the `iter_env`, see if we can extract features representing general market trend, then switch/ensemble different models. Especially when the market is in volatile, the strategy in general needs to be more conservative.
-  `feature_64` may be used as time to build CV strategy. Ref: [https://www.kaggle.com/c/jane-street-market-prediction/discussion/202253](https://www.kaggle.com/c/jane-street-market-prediction/discussion/202253)
- Number of trades `ts_id` might be related to volatility on that day: [https://www.kaggle.com/c/jane-street-market-prediction/discussion/201930#1125847](https://www.kaggle.com/c/jane-street-market-prediction/discussion/201930#1125847)
- Another notebook on volatility (this one we should use its info at dicretion): [https://www.kaggle.com/charlesleahy/identifying-volatility-features](https://www.kaggle.com/charlesleahy/identifying-volatility-features)
- Two-Sigma's stock prediction competition (with a similar format) winning solution of 5th place: [https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd](https://medium.com/kaggle-blog/two-sigma-financial-modeling-code-competition-5th-place-winners-interview-team-best-fitting-279a493c76bd)
- The data in the actual `test` is disjoint from the `train` (confirmed by the host at [https://www.kaggle.com/c/jane-street-market-prediction/discussion/199551](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199551))


# TO-DO:
- [ ] EDA
- [x] A simple starter
- [ ] Stable CV-LB strategy
- [ ] Writing a simple `iter_env` simulator

# NN models
## Autoencoder
- Kaggle Notebook: [https://www.kaggle.com/ztyreg/fork-of-s8900-ts](https://www.kaggle.com/ztyreg/fork-of-s8900-ts)
- Local Notebook: TBD
- Score: 8358.763
- Submission time: ~2 hours (CPU)

## AE+MLP+prediction cache
- Attempt 0.1: simply saving `pred_df.copy()` and using `pd.concat` is way too slow (7-8 iteration/s << 45 which is the current starter's). TO-DO: add a class so that prediction is a function under this class, model outputs to give more information, and some objects "depicting" the current market volatility. 


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
