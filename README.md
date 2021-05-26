# Playground for Jane Street Market Prediction Competition on Kaggle

# Introduction
Jane Street hosted a code competition of predicting the stock market (Feb 2021 to Aug 2021) using the past high frequency trading data (2 years of data before 2018?) on Kaggle: [https://www.kaggle.com/c/jane-street-market-prediction](https://www.kaggle.com/c/jane-street-market-prediction). 
The training data provided contain 500 days of high frequency trading data, total 2.4 million rows. The public leaderboard data contain 1 year of high frequency trading data from some time before Aug 2020 and up to that. The private ranges from a random time from July/Aug 2020 up to Aug 2021 (it was March 2021 as of the time of writings).  This training dataset contains an anonymized set of features, `feature_{0...129}`, representing real stock market data. Each row in the dataset represents a trading opportunity.

This is a code competition in that we have to prepare a pipeline of models that can do inference 1 trading opportunity at a time (no peaking into the future) subject to the inference API on Kaggle, and this submission should be able to perform the inference for 1.1 million samples in under 5 hours on cloud.
For each row, we will be predicting an action value: 1 to make the trade and 0 to pass on it. Each trade has an associated `weight` and `resp`, which together represents a return on the trade. The date column is an integer which represents the day of the trade, while `ts_id` represents a time ordering.

# Team: Semper Augustus
[Shuhao Cao](https://scaomath.github.io), [Carl McBride Ellis](http://www.sklogwiki.org/SklogWiki/carlmcbride.html), [Ethan Zheng](https://www.tzheng.org)

### Model performance on live stock market data update

| Date of LB |       Ranking        | Overfit Ensemble (OE) | OE delta | Local Best CV (LBC) | LBC delta |
|:----------:|:-------------------: |:---------------------:|:--------:|---------------------|:---------:|
|   Mar  5   | 99/4245, top 2.33%   |       4790.458        |          |       4541.474      |           |
|   Mar 17   | 75/4245, top 1.77%   |       5153.324        |   +363   |       4952.939      | +411      |
|   Mar 31   | 252/4245,top 5.93%   |       3934.002        |   -1219  |       3849.940      | -1103     |
|   Apr 14   | 260/4245,top 6.12%   |       3999.195        |   +65    |       4010.201      | +160      |
|   Apr 29   | 252/4245,top 5.93%   |       3843.239        |   -156   |       3889.275      | -121      |
|   May 12   | 152/4245,top 3.58%   |       4506.561        |   +663   |       4493.300      | +604      |

# Final submissions

## Data preparation

0. All data: only drop the two partial days and the two <2k `ts_id` days (done first).
1. `fillna()` uses the past day mean including all weight zero rows for every feature. 
2. ~~Most common values `fillna` for spike features rows.~~ (not any more after categorical embedding)
4. Smoother data: aside from 1, query `day > 85`, ~~drop `ts_id` > 9000 days~~(decreases CV by a margin so still included), dropping of data before day 85 is covered in Carl's EDA: [
Jane Street: EDA of day 0 and feature importance](https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance). 
5. Final training uses only `weight > 0` rows, ~~with a randomly selected 40% of weight zero rows' weight being replaced by 1e-7 to reduce overfitting~~ (reduces CV so discarded).
6. ~~A new de-noised target is generated with all five targets~~ (CV too good but leaderboard bad).

## Models
- (PT) PyTorch baseline with the skip connection mechanics, around 400k parameters, fast inference. Easy to get overfit.
- (S) Carl found that some features have an extremely high number of common values. Based on close inspection. I have a conjecture that they are certain categorical features' embedding. So this model is designed to add an embedding block for these features. Also with the skip connection mechanics, around 300k
parameters, best local CV and best single model leaderboard score.
- (AE) Tensorflow implementation of an autoencoder + a small MLP net with skip connection in the first layer. Small net. Currently the best scored public ones with a serious CV using 3 folds ensemble.
- (TF) Tensorflow Residual MLP using a filtering layer with high dropout rates to filter out hand-picked unimportant features suggested by Carl. 
- ~~(TF overfit) the infamous overfit model with a 1111 seed.~~ (we decided to exclude this one in the final submission)

## Train

### Validation score
Instead of the common accuracy or area-under-curve metrics for the classification problem, this competition is evaluated on a utility score. 

For each date $i$, we define: for `r` representing the `resp` (response), `w` representing the `weight`, and `a` representing the `action` (1 for taking the trade, 0s for pass):
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20p_i%20%3D%20%5Csum_%7Bj%7D%20w_%7Bij%7D%20r_%7Bij%7D%20a_%7Bij%7D">
</p>

Then it is summed up to 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20t%20%3D%20%5Cfrac%7B%5Csum%20p_i%20%7D%7B%5Csqrt%7B%5Csum%20p_i%5E2%7D%7D%20*%20%5Csqrt%7B%5Cfrac%7B250%7D%7B%7Ci%7C%7D%7D%2C">
</p>

Finally the utility is computed by:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20u%20%3D%20%5Cmin(%5Cmax(t%2C0)%2C%206)%20%20%5Csum_i%20p_i.">
</p>

Essentially, without considering some real market constraint, when every `p_i` become positive, this is to maximize 

<p align= "center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20%5Cleft(%5Csum_i%20p_i%5Cright)%5E2%20%5Ccdot%20%5Cleft(%20%0A%20%5Csum_i%20p_i%5E2%5Cright)%5E%7B-1%7D">
</p>
which we will use to construct a fine tuner for trained models.

### Train-validation strategy
A grouped validation strategy based on a total of 100 days as validation, a 10-day gap between the last day of train and the first of valid, three folds. The gap is due to the speculation of certain features being the moving average of certain metrics for the tradings.
```python
splits = {
          'train_days': (range(0,457), range(0,424), range(0,391)),
          'valid_days': (range(467, 500), range(434, 466), range(401, 433)),
          }
```

### Training of different models
1. Volatile models: all data with only `resp`, `resp_3`, `resp_4` as targets.
2. Smoother models: smoother data with all five `resp`s.
~~3. De-noised models: smoother data with all five `resp`s + a de-noised target~~.
4. Optimizer is simply Adam with a cosine annealing scheduler that allow warm restarts. Rectified Adam for tensorflow models.
5. During training of torch models, a fine-tuning regularizer is applied each 10 epochs to maximize the utility function by choosing action being the sigmoid of the outputs (Only for torch models, I do not know how to incorporate this in `tensorflow` training, as tensorflow's custom loss function is not that straightforward to keep track of extra inputs between batches).

## Submissions
1. Local best CV ones within a several-seeded bag. Final models: a set of `3(S) + 3(PT) + 3(AE) + 1(TF)`  for both smooth and volatile data.
2. ~~Trained with all data using the “public leaderboard as CV” epochs determined earlier, plus the infamous tensorflow seed 1111 overfit model. The validation for this submission is based on the variation of the utility score in all train data among all 25-day non-overlapping spans.~~
3. As our designated submission timed out... due to my poor judgement on the number of models to ensemble, we decided to choose an overfit model using the first pipeline.

## Inference 
1. CPU inference because the submission is CPU-bounded rather GPU. Torch models are usually faster than TF, TF models with `numba` backend enabled. (Update Feb 23 after the competition ended) I found that GPU inference became faster than CPU as more Tensorflow-based models are incorporated in the pipeline.
2. (Main contribution of Semper Augustus) Use `feature_64`'s [average gradient (a scaled version of $\arcsin (t)$) suggest by Carl](https://www.kaggle.com/c/jane-street-market-prediction/discussion/208013#1135364), and the number of trades in the previous day as a criterion to determine the models to include. Reference: [slope test of the past day class by Ethan and iter_cv simulation written by Shuhao](https://www.kaggle.com/ztyreg/validate-busy-prediction), [slope validation](https://www.kaggle.com/ztyreg/validate-busy-prediction)
3. Blending is always concatenating models in a bag then taking the middle 60%'s
average (median if only 3 models), then concatenating again to take the middle 60% average (50% if a day is busy). For example, if we have `5 (PT) + 3 (AE) + 1 (TF)`, then `5 (PT)`'s predictions
are concatenated and averaged along `axis 0` with the middle three, and `(AE)` submissions are taken the median. Lastly, the subs are concatenated again to take the middle 9 entries (15 total).
4. Regular days: `3 (P)`, `3 (S)` ~~with denoised target~~, `3 (AE)`, and 1
`(TF)` trained on the smoother models.
1. Busy days: above models trained on all data.

----
### Below were the notes and tries before we orchestrated our final solution.


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
- Attempt 0.1: simply saving `pred_df.copy()` and using `pd.concat` is way too slow (7-8 iteration/s << 45 which is the current starter's). 
- TO-DO: add a class so that prediction is a function under this class, model outputs to give more information, and some objects "depicting" the current market volatility. 

## A new Residual+MLP model
- The key is to train using the actual `resp` columns as target, and when doing the inference, apply the sigmoid function to the output (why `BCEwLogits` performs better than `CrossEntropy`???).
- Set up the baseline training, adding a 16-target model (using various sums between the `resp` columns).
- Tested the sensitivity of seeds to the CV vs public leaderboard. Bigger model in general is less sensitive than smaller models (esp the seed 1111 overfit model).
- A local-public LB stable training strategy: RAdam/Adam with cosine annealing scheduler, utility function regularizer finetuning every 10 epochs with a `1e-3*lr` learning rate, 1 or 2 denoised targets added, 50% median average ensembling. 
- Feature neutralization might not fit the iteration speed needed for the inference.

## Validation scores

#### ResNet (TF), two features group, regular days
| Fold   |      Seed   |  Score |
|--------|:-----------:|------: |
| 0      |  1127802    | 1621.86 |
| 1      |  1127802    | 1080.24 |
| 1      |  792734     | 1221.17 |
| 2      |  1127802    | 80.85   |
| 2      |  97275      | 146.31  |
| 0      |  157157    | 1554.01|
| 1      |  157157    | 1273.48 |
| 2      |  157157    | 19.76  |

#### ResNet (TF), two features group, volatile days
| Fold   |      Seed   |  Score |
|--------|:-----------:|------: |
| 0      |  1127802    | 1640.27 |
| 1      |  1127802    | 1054.42 |
| 2      |  1127802    | 45.15  |
| 0      |  157157    | 1563.25 |
| 1      |  157157    |  1253.98 |
| 2      |  157157    | 11.14  |
| 0      |  745273    | 1511.12 |
| 1      |  962656  |  0.01 |
| 0      |  5567273    |  1457.13 |
| 1      |  123835    | 1290.73 |
| 2      |  676656    | 34.38  |

#### ResNet+spike (TF+S), three features group, regular days (too slow for inference not going into the final sub pipeline)
| Fold   |      Seed   |  Score |
|--------|:-----------:|------: |
| 0      |  1127802    | 1417.43 |
| 1      |  1127802    | 1082.22 |
| 2      |  1127802    | 59.87   |
| 2      |  802        | 175.96 |

#### (AE) regular days (only first two folds are used due to time constraint)
| Fold   |      Seed   |  Score |
|--------|:-----------:|------: |
| 0      |  692874     | 1413.37 |
| 0      |  1127802    | 1552.13 |
| 1      |  692874     | 1037.59 |
| 1      |  1127802    | 1209.71 |
| 2      |  692874     | 144.69 |
| 2      |  1127802    | 144.29  |
| 0      |  157157    | 1529.70 |
| 1      |  157157    | 1052.70 |
| 2      |  157157    | 402.80  |

#### (AE) volatile days (only first two folds are used due to time constraint)
| Fold   |      Seed   |  Score |
|--------|:-----------:|------: |
| 0      |  969725     | 1485.01 |
| 0      |  1127802    | 1672.50 |
| 0      |  618734     | 1623.88 |
| 0      |  283467     | 1670.67 |
| 1      |  969725     | 1284.02 |
| 1      |  1127802    | 1347.90 |
| 1      |  618734     | 969.63 |
| 1      |  283467    | 1006.84 |
| 2      |  969725     | 0.83 |
| 2      |  1127802    | 0.26  |
| 2      |  618734     | 0    |
| 2      |  283467     | 49.79 |

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
