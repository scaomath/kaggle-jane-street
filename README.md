# Playground for Jane Street Market Prediction Competition on Kaggle

# Team: Semper Augustus
[Shuhao Cao](https://scaomath.github.io), [Carl McBride Ellis](http://www.sklogwiki.org/SklogWiki/carlmcbride.html), [Ethan Zheng](https://www.tzheng.org)

# Final submissions

## Data preparation
The data contain 500 days of high frequency trading data from Jane Street, total 2.4 million rows.

0. All data: only drop the two partial days and the two <2k `ts_id` days (done first).
1. `fillna()` past day mean including all weight zero rows. 
2. ~~Most common values `fillna` for spike features rows.~~ (not any more after categorical embedding)
4. Smoother data: aside from 1, query day > 85, drop `ts_id` > 9000 days.
5. Final training uses only `weight > 0` rows, ~~with a randomly selected 40% of weight zero rows' weight being replaced by 1e-7 to reduce overfitting~~ (reduces CV so discarded).
6. ~~A new de-noised target is generated with all five targets~~ (CV too good but leaderboard bad).

## Models
- (PT) PyTorch baseline with the skip connection mechanics, around 400k parameters, fast inference. Easy to get overfit.
- (S) Carl found that some features have an extremely high number of common values. Based on close inspection. I have a conjecture that they are certain categorical features' embedding. So this model is designed to add an embedding block for these features. Also with the skip connection mechanics, around 300k
parameters, best local CV and best single model leaderboard score.
- (AE) Tensorflow implementation of an autoencoder + a small MLP net with skip connection in the first layer. Small net. Currently the best scored public ones with a serious CV using 3 folds ensemble.
- (TF) Tensorflow Residual MLP using a filtering layer with high dropout rates to filter out hand-picked unimportant features suggested by Carl.
- (TF overfit) the infamous overfit model with a 1111 seed.

## Train

### Train-validation splits
A grouped validation strategy based on a total of 100 days as validation, a 10-day gap between the last day of train and the first of valid, three folds.
```python
splits = {
          'train_days': (range(0,457), range(0,424), range(0,391)),
          'valid_days': (range(467, 500), range(434, 466), range(401, 433)),
          }
```

1. Volatile models: all data with only `resp`, `resp_3`, `resp_4` as targets.
2. Smoother models: smoother data with all five `resp`s.
~~3. De-noised models: smoother data with all five `resp`s + a de-noised target~~.
4. Optimizer is simply Adam with a cosine annealing scheduler that allow warm restarts. Rectified Adam for tensorflow models.
5. During training of torch models, a fine-tuning regularizer is applied each 10 epochs to maximize the utility function by choosing action being the sigmoid of the outputs (Only for torch models, I do not know how to incorporate this in `tensorflow` training, as tensorflow's custom loss function is not that straightforward to keep track of extra inputs between batches).

## Submissions
1. Local best CV ones within a three seeds bag. Final models: a set of `3(S) + 3(PT) + 3(AE) + 1(TF)`  for both smooth and volatile data.
2. Trained with all data using the “public leaderboard as CV” epochs determined earlier, plus the infamous tensorflow seed 1111 overfit model.

## Inference 
1. CPU inference because the submission is CPU-bounded rather GPU. Torch models are usually faster than TF, TF models with `numba` backend enabled.
2. (Main contribution of Semper Augustus) Use `feature_64`'s [average gradient (a scaled version of $\arcsin (t)$) suggest by Carl](https://www.kaggle.com/c/jane-street-market-prediction/discussion/208013#1135364), and the number of trades in the previous day as a criterion to determine the models to include. Reference: [slope test of the past day class by Ethan and iter_cv simulation written by Shuhao](https://www.kaggle.com/ztyreg/validate-busy-prediction), [slope validation](https://www.kaggle.com/ztyreg/validate-busy-prediction)
3. Blending is always concatenating models in a bag then taking the middle 60%'s
average (median if only 3 models), then concatenating again to take the middle 60% average (50% if a day is busy). For example, if we have `5 (PT) + 3 (AE) + 1 (TF)`, then `5 (PT)`'s predictions
are concatenated and averaged along `axis 0` with the middle three, and `(AE)` submissions are taken the median. Lastly, the subs are concatenated again to take the middle 9 entries (15 total).
4. Regular days: `3 (P)`, `3 (S)` ~~with denoised target~~, `3 (AE)`, and 1
`(TF)` trained on the smoother models.
1. Busy days: above models trained on all data.



# Things to try for the final submission:
- [x] Simple EDA.
- [x] A simple starter.
- [x] Stable CV-LB strategy (Updated Jan 22, now I think this is somehow impossible; updated Feb 12, certain correlation between the LB and the denoised target utility-finetuning around 70 epochs of ADAM).
- [x] Writing a simple `iter_env` simulator.
- [x] Testing a moving average `fillna()` strategy in both train and inference pipeline.
- [x] Testing a past mean `fillna()`, fill the NaN using the mean only from prior day data, no intraday data.
- [x] Using the `iter_env` simulator to test the impact of different threshold: 0.502 or 0.498 can be both better than 0.5? Need an explanation...
- [ ] A table compiling what features will be using `ffill`, previous day mean, overall mean, etc (maybe not necessary?).
- [x] Trading frequency can be determined by number of trades per day, store this in a cache to choose model.
- [ ] Using `feature_0` to choose models, and/or threshold (based on `feature_0`'s previous day count?).
- [x] Using rolling mean/exponential weighted mean of previous days as input/fillna, working out a submission pipeline.
- [x] Implement a regularizer using the utility function.
- [x] Train with all weights (maybe making `weight==0` rows' weights to certain small number `1e-5`), then train with all positive `weight` rows (slightly better public leaderboard).
- [x] Train with a weighted cross entropy loss, the weight is $\ln(1+w)$; the local CV became better but public leaderboard became worse. 
- [x] Adding one or multiple de-noised targets by removing the eigenvalues of the covariance matrix.
- [x] Train models including the first 85 days but excluding outlier days (high volatility days). For low volatile days, use the denoised models (?).
- [x] Use public LB to do a variance test to determine whether the seed 1111 overfitting model can be used to do final submission. (weighted by 8 due to the total days factor) Public test 0-25:2565, 25-50:4131, 50-75:3156, 75-100:743; std=1234.
- [x] Testing the correlation between, for example `feature 3`'s exponential weighted mean and `resp`s columns (or other transforms) (update Feb 21: both exponential moving averaging and windowed rolling mean do not help the CV).


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
- Carl's observation: huge spike in the histogram of features 3,4,6,19,20,22,38 etc, also lurking on the far left side of features 71, 85, 87, 92, 97, 105, 127 and 129. A smaller spike is seen for feature 116.

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
