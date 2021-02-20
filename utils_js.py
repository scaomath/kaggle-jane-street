from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import numpy as np
import pandas as pd

from numba import njit
import os
import math
import torch
from torch.optim import Optimizer
HOME = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = HOME+'/models/'
DATA_DIR = HOME+'/data/'
from utils import *

import itertools as itt
import numbers

from abc import abstractmethod
from typing import Iterable, Tuple, List


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

'''
Original Code from https://github.com/sam31415/timeseriescv
Modified by treename@Kaggle in https://www.kaggle.com/treename/janestreet-cv-method-combpurgedkfoldcv
'''

class BaseTimeSeriesCrossValidator:
    """
    Abstract class for time series cross-validation.
    Time series cross-validation requires each sample has a prediction time pred_time, at which the features are used to
    predict the response, and an evaluation time eval_time, at which the response is known and the error can be
    computed. Importantly, it means that unlike in standard sklearn cross-validation, the samples X, response y,
    pred_times and eval_times must all be pandas dataframe/series having the same index. It is also assumed that the
    samples are time-ordered with respect to the prediction time (i.e. pred_times is non-decreasing).
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    """
    def __init__(self, n_splits=10):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(f"The number of folds must be of Integral type. {n_splits} of type {type(n_splits)}"
                             f" was passed.")
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting n_splits = 2 "
                             f"or more, got n_splits = {n_splits}.")
        self.n_splits = n_splits
        self.pred_times = None
        self.eval_times = None
        self.indices = None

    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None):
        if not isinstance(X, pd.DataFrame) and not isinstance(X, pd.Series):
            raise ValueError('X should be a pandas DataFrame/Series.')
        if not isinstance(y, pd.Series) and y is not None:
            raise ValueError('y should be a pandas Series.')
        if not isinstance(pred_times, pd.Series):
            raise ValueError('pred_times should be a pandas Series.')
        if not isinstance(eval_times, pd.Series):
            raise ValueError('eval_times should be a pandas Series.')
        if y is not None and (X.index == y.index).sum() != len(y):
            raise ValueError('X and y must have the same index')
        if (X.index == pred_times.index).sum() != len(pred_times):
            raise ValueError('X and pred_times must have the same index')
        if (X.index == eval_times.index).sum() != len(eval_times):
            raise ValueError('X and eval_times must have the same index')

        self.pred_times = pred_times
        self.eval_times = eval_times
        self.indices = np.arange(X.shape[0])


class CombPurgedKFoldCV(BaseTimeSeriesCrossValidator):
    """
    Purged and embargoed combinatorial cross-validation
    As described in Advances in financial machine learning, Marcos Lopez de Prado, 2018.
    The samples are decomposed into n_splits folds containing equal numbers of samples, without shuffling. In each cross
    validation round, n_test_splits folds are used as the test set, while the other folds are used as the train set.
    There are as many rounds as n_test_splits folds among the n_splits folds.
    Each sample should be tagged with a prediction time pred_time and an evaluation time eval_time. The split is such
    that the intervals [pred_times, eval_times] associated to samples in the train and test set do not overlap. (The
    overlapping samples are dropped.) In addition, an "embargo" period is defined, giving the minimal time between an
    evaluation time in the test set and a prediction time in the training set. This is to avoid, in the presence of
    temporal correlation, a contamination of the test set by the train set.
    Parameters
    ----------
    n_splits : int, default=10
        Number of folds. Must be at least 2.
    n_test_splits : int, default=2
        Number of folds used in the test set. Must be at least 1.
    embargo_td : pd.Timedelta, default=0
        Embargo period (see explanations above).
    """
    def __init__(self, n_splits=10, n_test_splits=2, embargo_td=0):
        super().__init__(n_splits)
        if not isinstance(n_test_splits, numbers.Integral):
            raise ValueError(f"The number of test folds must be of Integral type. {n_test_splits} of type "
                             f"{type(n_test_splits)} was passed.")
        n_test_splits = int(n_test_splits)
        if n_test_splits <= 0 or n_test_splits > self.n_splits - 1:
            raise ValueError(f"K-fold cross-validation requires at least one train/test split by setting "
                             f"n_test_splits between 1 and n_splits - 1, got n_test_splits = {n_test_splits}.")
        self.n_test_splits = n_test_splits

        if embargo_td < 0:
            raise ValueError(f"The embargo time should be positive, got embargo = {embargo_td}.")
        self.embargo_td = embargo_td

    def split(self, X: pd.DataFrame, y: pd.Series = None,
              pred_times: pd.Series = None, eval_times: pd.Series = None) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield the indices of the train and test sets.
        Although the samples are passed in the form of a pandas dataframe, the indices returned are position indices,
        not labels.
        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features), required
            Samples. Only used to extract n_samples.
        y : pd.Series, not used, inherited from _BaseKFold
        pred_times : pd.Series, shape (n_samples,), required
            Times at which predictions are made. pred_times.index has to coincide with X.index.
        eval_times : pd.Series, shape (n_samples,), required
            Times at which the response becomes available and the error can be computed. eval_times.index has to
            coincide with X.index.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        """
        super().split(X, y, pred_times, eval_times)

        # Fold boundaries
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(self.indices, self.n_splits)]
        # List of all combinations of n_test_splits folds selected to become test sets
        selected_fold_bounds = list(itt.combinations(fold_bounds, self.n_test_splits))
        
        # In order for the first round to have its whole test set at the end of the dataset
        selected_fold_bounds.reverse()

        for fold_bound_list in selected_fold_bounds:
            # Computes the bounds of the test set, and the corresponding indices
            test_fold_bounds, test_indices = self.compute_test_set(fold_bound_list)
            # Computes the train set indices
            train_indices = self.compute_train_set(test_fold_bounds, test_indices)

            yield train_indices, test_indices

    def compute_train_set(self, test_fold_bounds: List[Tuple[int, int]], test_indices: np.ndarray) -> np.ndarray:
        """
        Compute the position indices of samples in the train set.
        Parameters
        ----------
        test_fold_bounds : List of tuples of position indices
            Each tuple records the bounds of a block of indices in the test set.
        test_indices : np.ndarray
            A numpy array containing all the indices in the test set.
        Returns
        -------
        train_indices: np.ndarray
            A numpy array containing all the indices in the train set.
        """
        # As a first approximation, the train set is the complement of the test set
        train_indices = np.setdiff1d(self.indices, test_indices)
        # But we now have to purge and embargo
        for test_fold_start, test_fold_end in test_fold_bounds:
            # Purge
            train_indices = purge(self, train_indices, test_fold_start, test_fold_end)
            # Embargo
            train_indices = embargo(self, train_indices, test_indices, test_fold_end)
        return train_indices

    def compute_test_set(self, fold_bound_list: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Compute the indices of the samples in the test set.
        Parameters
        ----------
        fold_bound_list: List of tuples of position indices
            Each tuple records the bounds of the folds belonging to the test set.
        Returns
        -------
        test_fold_bounds: List of tuples of position indices
            Like fold_bound_list, but with the neighboring folds in the test set merged.
        test_indices: np.ndarray
            A numpy array containing the test indices.
        """
        test_indices = np.empty(0)
        test_fold_bounds = []
        for fold_start, fold_end in fold_bound_list:
            # Records the boundaries of the current test split
            if not test_fold_bounds or fold_start != test_fold_bounds[-1][-1]:
                test_fold_bounds.append((fold_start, fold_end))
            # If the current test split is contiguous to the previous one, simply updates the endpoint
            elif fold_start == test_fold_bounds[-1][-1]:
                test_fold_bounds[-1] = (test_fold_bounds[-1][0], fold_end)
            test_indices = np.union1d(test_indices, self.indices[fold_start:fold_end]).astype(int)
        return test_fold_bounds, test_indices


def compute_fold_bounds(cv: BaseTimeSeriesCrossValidator, split_by_time: bool) -> List[int]:
    """
    Compute a list containing the fold (left) boundaries.
    Parameters
    ----------
    cv: BaseTimeSeriesCrossValidator
        Cross-validation object for which the bounds need to be computed.
    split_by_time: bool
        If False, the folds contain an (approximately) equal number of samples. If True, the folds span identical
        time intervals.
    """
    if split_by_time:
        full_time_span = cv.pred_times.max() - cv.pred_times.min()
        fold_time_span = full_time_span / cv.n_splits
        fold_bounds_times = [cv.pred_times.iloc[0] + fold_time_span * n for n in range(cv.n_splits)]
        return cv.pred_times.searchsorted(fold_bounds_times)
    else:
        return [fold[0] for fold in np.array_split(cv.indices, cv.n_splits)]


def embargo(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
            test_indices: np.ndarray, test_fold_end: int) -> np.ndarray:
    """
    Apply the embargo procedure to part of the train set.
    This amounts to dropping the train set samples whose prediction time occurs within self.embargo_dt of the test
    set sample evaluation times. This method applies the embargo only to the part of the training set immediately
    following the end of the test set determined by test_fold_end.
    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times, cv.embargo_dt and cv.indices.
    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.
    test_indices : np.ndarray
        A numpy array containing all the indices of the samples in the test set.
    test_fold_end : int
        Index corresponding to the end of a test set block.
    Returns
    -------
    train_indices: np.ndarray
        The same array, with the indices subject to embargo removed.
    """
    if not hasattr(cv, 'embargo_td'):
        raise ValueError("The passed cross-validation object should have a member cv.embargo_td defining the embargo"
                         "time.")
    last_test_eval_time = cv.eval_times.iloc[cv.indices[:test_fold_end]].max()
    min_train_index = len(cv.pred_times[cv.pred_times <= last_test_eval_time + cv.embargo_td])
    if min_train_index < cv.indices.shape[0]:
        allowed_indices = np.concatenate((cv.indices[:test_fold_end], cv.indices[min_train_index:]))
        train_indices = np.intersect1d(train_indices, allowed_indices)
    return train_indices


def purge(cv: BaseTimeSeriesCrossValidator, train_indices: np.ndarray,
          test_fold_start: int, test_fold_end: int) -> np.ndarray:
    """
    Purge part of the train set.
    Given a left boundary index test_fold_start of the test set, this method removes from the train set all the
    samples whose evaluation time is posterior to the prediction time of the first test sample after the boundary.
    Parameters
    ----------
    cv: Cross-validation class
        Needs to have the attributes cv.pred_times, cv.eval_times and cv.indices.
    train_indices: np.ndarray
        A numpy array containing all the indices of the samples currently included in the train set.
    test_fold_start : int
        Index corresponding to the start of a test set block.
    test_fold_end : int
        Index corresponding to the end of the same test set block.
    Returns
    -------
    train_indices: np.ndarray
        A numpy array containing the train indices purged at test_fold_start.
    """
    time_test_fold_start = cv.pred_times.iloc[test_fold_start]
    # The train indices before the start of the test fold, purged.
    train_indices_1 = np.intersect1d(train_indices, cv.indices[cv.eval_times < time_test_fold_start])
    # The train indices after the end of the test fold.
    train_indices_2 = np.intersect1d(train_indices, cv.indices[test_fold_end:])
    

    return np.concatenate((train_indices_1, train_indices_2))

class Lookahead(Optimizer):
    r'''Implements Lookahead optimizer.
    https://github.com/nachiket273/lookahead_pytorch
    It's been proposed in paper: Lookahead Optimizer: k steps forward, 1 step back
    (https://arxiv.org/pdf/1907.08610.pdf)
    Args:
        optimizer: The optimizer object used in inner loop for fast weight updates.
        alpha:     The learning rate for slow weight update.
                   Default: 0.5
        k:         Number of iterations of fast weights updates before updating slow
                   weights.
                   Default: 5
    Example:
        > optim = Lookahead(optimizer)
        > optim = Lookahead(optimizer, alpha=0.6, k=10)
    '''
    def __init__(self, optimizer, alpha=0.5, k=5):
        assert(0.0 <= alpha <= 1.0)
        assert(k >= 1)
        self.optimizer = optimizer
        self.alpha = alpha
        self.k = k
        self.k_counter = 0
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.slow_weights = [[param.clone().detach() for param in group['params']] for group in self.param_groups]
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.k_counter += 1
        if self.k_counter >= self.k:
            for group, slow_weight in zip(self.param_groups, self.slow_weights):
                for param, weight in zip(group['params'], slow_weight):
                    weight.data.add_(self.alpha, (param.data - weight.data))
                    param.data.copy_(weight.data)
            self.k_counter = 0
        return loss

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'k': self.k,
            'k_counter': self.k_counter
        }

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

class LookaheadOld(Optimizer):
    '''
    https://github.com/alphadl/lookahead.pytorch
    '''
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class RAdam(Optimizer):
    r"""Implements RAdam algorithm.
    https://github.com/nachiket273/lookahead_pytorch
    It has been proposed in `ON THE VARIANCE OF THE ADAPTIVE LEARNING
    RATE AND BEYOND(https://arxiv.org/pdf/1908.03265.pdf)`_.
    
    Arguments:
        params (iterable):      iterable of parameters to optimize or dicts defining
                                parameter groups
        lr (float, optional):   learning rate (default: 1e-3)
        betas (Tuple[float, float], optional):  coefficients used for computing
                                                running averages of gradient and 
                                                its square (default: (0.9, 0.999))
        eps (float, optional):  term added to the denominator to improve
                                numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional):    whether to use the AMSGrad variant of this
                                        algorithm from the paper `On the Convergence 
                                        of Adam and Beyond`_(default: False)
        
        sma_thresh:             simple moving average threshold.
                                Length till where the variance of adaptive lr is intracable.
                                Default: 4 (as per paper)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sma_thresh=4):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(RAdam, self).__init__(params, defaults)

        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.sma_thresh = sma_thresh

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                old = p.data.float()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                buffer = self.radam_buffer[int(state['step']%10)]

                if buffer[0] == state['step']:
                    sma_t, step_size = buffer[1], buffer[2]
                else:                 
                    sma_max_len = 2/(1-beta2) - 1  
                    beta2_t = beta2 ** state['step']
                    sma_t = sma_max_len - 2 * state['step'] * beta2_t /(1 - beta2_t)
                    buffer[0] = state['step']
                    buffer[1] = sma_t

                    if sma_t > self.sma_thresh :
                        rt = math.sqrt(((sma_t - 4) * (sma_t - 2) * sma_max_len)/((sma_max_len -4) * (sma_max_len - 2) * sma_t))
                        step_size = group['lr'] * rt * math.sqrt((1 - beta2_t)) / (1 - beta1 ** state['step'])                      
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])                        
                    buffer[2] = step_size

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], old)

                if sma_t > self.sma_thresh :
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p.data.add_(-step_size, exp_avg)

        return loss





def preprocess_base(df, drop_weight=True):
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

def add_denoised_target(train_df, num_dn_target=1):
    for i in range(num_dn_target):
        target_denoised = pd.read_csv(os.path.join(DATA_DIR, f'target_dn_{i}.csv'))
        train_df = pd.concat([train_df, target_denoised], axis=1)
        train_df[f'action_dn_{i}'] = (train_df[f'resp_dn_{i}']>0).astype(int)
    return train_df

## preprocess for torch model
def preprocess_pt(train_file, day_start=86, 
                  day_split=475, drop_days=None,
                  drop_zero_weight=True, zero_weight_thresh=1e-6, 
                  denoised_resp=False, num_dn_target=1):
    try:
        train = pd.read_parquet(train_file)
    except:
        train = pd.read_feather(train_file)

    if drop_days:
        train = train.query(f'date not in {drop_days}').reset_index(drop = True)
    
    if denoised_resp:
        train = add_denoised_target(train, num_dn_target=num_dn_target)
        
    train = train.loc[train.date >= day_start].reset_index(drop=True)

    if drop_zero_weight:
        train = train[train['weight'] > 0].reset_index(drop = True)
    elif drop_zero_weight==False and zero_weight_thresh is not None:
        len_zero_weight = (train['weight'] == 0).sum()

        # train[['weight']] = train[['weight']].clip(zero_weight_thresh)
        
        noise = zero_weight_thresh*np.random.rand(len_zero_weight)
        train.loc[train['weight'] == 0, ['weight']] = noise.clip(0)

    # vanilla actions based on resp
    train['action'] = (train['resp'] > 0).astype(int)
    for c in range(1,5):
        train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(int)

    train.fillna(train.mean(),inplace=True)
    
    train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
    train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)

    if day_split is not None:
        valid = train.loc[train.date >= day_split].reset_index(drop=True)
        train = train.loc[train.date < day_split].reset_index(drop=True)
        return train, valid
    else:
        return train

def preprocess_final(train_file, day_start=86, drop_days=None,
                  drop_zero_weight=True, zero_weight_thresh=1e-7, 
                  training_days=None, valid_days=None,
                  denoised_resp=False, num_dn_target=1):
    try:
        train = pd.read_parquet(train_file)
    except:
        train = pd.read_feather(train_file)
    train = train.loc[train.date >= day_start].reset_index(drop=True)

    if drop_days:
        train = train.query(f'date not in {drop_days}').reset_index(drop = True)
    
    if denoised_resp:
        train = add_denoised_target(train, num_dn_target=num_dn_target)

    if drop_zero_weight:
        train = train[train['weight'] > 0].reset_index(drop = True)
    elif drop_zero_weight==False and zero_weight_thresh is not None:
        train[['weight']] = train[['weight']].clip(zero_weight_thresh)

    # vanilla actions based on resp
    train['action'] = (train['resp'] > 0).astype(int)
    for c in range(1,5):
        train['action_'+str(c)] = (train['resp_'+str(c)] > 0).astype(int)

    # train.fillna(train.mean(),inplace=True)
    
    train['cross_41_42_43'] = train['feature_41'] + train['feature_42'] + train['feature_43']
    train['cross_1_2'] = train['feature_1'] / (train['feature_2'] + 1e-5)

    if valid_days is not None and training_days is not None:
        valid = train.loc[train.date.isin(valid_days)].reset_index(drop=True)
        train = train.loc[train.date.isin(training_days)].reset_index(drop=True)
        return train, valid
    else:
        return train


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

# @njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


def median_avg(predictions, beta=0.7, axis=-1, debug=False):
    '''
    predictions should be of a vector shape (..., n_models)
    beta: if beta is 0.5, then the middle 50% will be averaged
    '''
    sorted_predictions = np.sort(predictions, axis=axis)
    n_model = sorted_predictions.shape[-1]
    mid_point = n_model//2+1
    n_avg = int(n_model*beta)
    to_avg = sorted_predictions.take(range(mid_point-n_avg//2-1,mid_point+n_avg//2), axis=axis)

    if debug:
        print('sorted_list',sorted_predictions)
        print('after_cut',sorted_predictions[..., mid_point-n_avg//2-1:mid_point+n_avg//2])

    return to_avg.mean(axis=axis)

#Designed to do all features at the same time, but Kaggle kernels are memory limited.
class NeutralizeTransform:
    '''
    Unfortunately too slow for submission API
    Reference: https://www.kaggle.com/snippsy/jane-street-densenet-neutralizing-features
    '''
    def __init__(self,proportion=1.0):
        self.proportion = proportion
    
    def fit(self,X,y):
        self.lms = []
        self.mean_exposure = np.mean(y,axis=0)
        self.y_shape = y.shape[-1]
        for x in X.T:
            scores = x.reshape((-1,1))
            exposures = y
            exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))
            
            transform = np.linalg.lstsq(exposures, scores, rcond=None)[0]
            self.lms.append(transform)
            
    def transform(self,X,y=None):
        out = []
        for i,transform in enumerate(self.lms):
            x = X[:,i]
            scores = x.reshape((-1,1))
            exposures = np.repeat(self.mean_exposure,len(x),axis=0).reshape((-1,self.y_shape))
            exposures = np.concatenate([exposures,np.array([np.mean(scores)] * len(exposures)).reshape((-1,1))],axis=1)
            correction = self.proportion * exposures.dot(transform)
            out.append(x - correction.ravel())
            
        return np.asarray(out).T
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X,y)



if __name__ == "__main__":
    HOME = os.path.dirname(os.path.abspath(__file__))
    with timer("Loading train parquet"):
        train_parquet = os.path.join(HOME,'data/train.parquet')
        train = pd.read_parquet(train_parquet)
    train['action'] = (train['resp'] > 0)
    for c in range(1,5):
        train['action'] = train['action'] & ((train['resp_'+str(c)] > 0))
    
    train = train.query('date > 296').reset_index(drop=True)

    date = train['date'].values
    weight = train['weight'].values
    resp = train['resp'].values
    action = train['action'].values

    print(f"Number of rows in train: {len(train)}")

    with timer("Numba", compact=True):
        utility_score_numba(date[:1], weight[:1], resp[:1], action[:1])
        print(utility_score_numba(date, weight, resp, action))

    with timer("numpy", compact=True): # fastest
        print(utility_score_bincount(date, weight, resp, action))

    with timer("loops", compact=True):
        print(utility_score_loop(date, weight, resp, action))

    with timer("Pandas", compact=True):
        print(utility_score_pandas(train, labels = ['action', 'resp', 'weight', 'date']))

    with timer("Pandas2", compact=True):
        print(utility_score_pandas2(train))