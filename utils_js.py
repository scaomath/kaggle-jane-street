from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import numpy as np
from numba import njit
import tensorflow as tf
from utils import *

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]



def preprocess(df, drop_weight=True):
    '''
    Only use day > 85 data
    Default: drop weight 0 trades
    '''
    features = [c for c in df.columns if 'feature' in c]
    resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']
    df = df[df['date'] > 85].reset_index(drop = True) 
    df.fillna(df.mean(),inplace=True)
    if drop_weight:
        df = df[df['weight'] > 0].reset_index(drop = True)  
    df['action'] = (df['resp'] > 0)
    for c in range(1,5):
        df['action'] = df['action'] & ((df['resp_'+str(c)] > 0))
    df['action'] = df['action'].astype('int')
    columns = ['date', 'ts_id', 'action']
    columns += [col for col in df.columns if col not in ['date', 'ts_id', 'action']]
    df = df[columns]
    return df


'''
Simulate the inference env of Kaggle
Utility function taken from https://www.kaggle.com/gogo827jz/jane-street-super-fast-utility-score-function
'''
def utility_score_loop(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.zeros(count_i)
    for i, day in enumerate(np.unique(date)):
        Pi[i] = np.sum(weight[date == day] * resp[date == day] * action[date == day])
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

def utility_score_bincount(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

@njit(fastmath = True)
def utility_score_numba(date, weight, resp, action):
    count_i = len(np.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = min(max(t, 0), 6) * np.sum(Pi)
    return u


def utility_score_pandas(df, labels='action,.r0,.weight,.date'.split(',')):
    """
    LDMTWO's pandas implementation
    Calculate utility score of a dataframe according to formulas defined at
    https://www.kaggle.com/c/jane-street-market-prediction/overview/evaluation
    """
    action,resp,weight,date = labels
    df = df.set_index(date)
    p = df[weight]  * df[resp] * df[action]
    p_i = p.groupby(date).sum()
    t = (p_i.sum() / np.sqrt((p_i**2).sum())) * (np.sqrt(250 / p_i.index.size))
    return np.clip(t,0,6) * p_i.sum()


def utility_score_pandas2(df):
    """
    Jorijn Jacko Smit's another pandas implementation
    Calculate utility score of a dataframe according to formulas defined at
    https://www.kaggle.com/c/jane-street-market-prediction/overview/evaluation
    """

    df['p'] = df['weight']  * df['resp'] * df['action']
    p_i = df.set_index('date')['p'].groupby('date').sum()
    t = (p_i.sum() / np.sqrt((p_i**2).sum())) * (np.sqrt(250 / p_i.index.size))
    return min(max(t, 0), 6) * p_i.sum()

@njit(fastmath = True)
def decision_threshold_optimisation(preds, date, weight, resp, low = 0, high = 1, bins = 100, eps = 1):
    opt_threshold = low
    gap = (high - low) / bins
    action = np.where(preds >= opt_threshold, 1, 0)
    opt_utility = utility_score_numba(date, weight, resp, action)
    for threshold in np.arange(low, high, gap):
        action = np.where(preds >= threshold, 1, 0)
        utility = utility_score_numba(date, weight, resp, action)
        if utility - opt_utility > eps:
            opt_threshold = threshold
            opt_utility = utility
    print(f'Optimal Decision Threshold:   {opt_threshold}')
    print(f'Optimal Utility Score:        {opt_utility}')
    return opt_threshold, opt_utility

@njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


def median_avg(predictions, beta=0.5, debug=False):
    '''
    predictions should be of a vector shape n_models
    beta: if beta is 0.5, then the middle 50% will be averaged
    '''
    sorted_predictions=np.sort(predictions)
    n_model=len(sorted_predictions)
    mid_point=n_model//2+1
    n_avg=int(n_model*beta)
    if debug:
        print('sorted_list',sorted_predictions)
        print('after_cut',sorted_predictions[mid_point-n_avg//2-1:mid_point+n_avg//2])
    to_avg=sorted_predictions[mid_point-n_avg//2-1:mid_point+n_avg//2]
    return np.mean(to_avg)

def create_mlp_tf(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)

    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )
    return model

if __name__ == "__main__":
    HOME = os.path.dirname(os.path.abspath(__file__))
    with timer("Loading train parquet"):
        train_parquet = os.path.join(HOME,'data/train.parquet')
        train = pd.read_parquet(train_parquet)
    train['action'] = (train['resp'] > 0)
    for c in range(1,5):
        train['action'] = train['action'] & ((train['resp_'+str(c)] > 0))
    
    date = train['date'].values
    weight = train['weight'].values
    resp = train['resp'].values
    action = train['action'].values



    with timer("Numba", compact=True):
        print(utility_score_numba(date, weight, resp, action))

    with timer("numpy", compact=True): # fastest
        print(utility_score_bincount(date, weight, resp, action))

    with timer("loops", compact=True):
        print(utility_score_loop(date, weight, resp, action))

    with timer("Pandas", compact=True):
        print(utility_score_pandas(train, labels = ['action', 'resp', 'weight', 'date']))

    with timer("Pandas2", compact=True):
        print(utility_score_pandas2(train))