import gc
import sys
import os
def add_sys_path():
    try:
        for f in ['/home/scao/anaconda3/lib/python3.8/lib-dynload',
                 '/home/scao/anaconda3/lib/python3.8/site-packages']:
            sys.path.append(f)
    except:
        RuntimeError
        print("Path not added")
add_sys_path()


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random as rd
from contextlib import contextmanager
from collections import defaultdict
from time import time
import matplotlib.pyplot as plt
from datetime import date
import math
import numpy as np
import pandas as pd
import psutil
import torch
import pickle
import seaborn as sns
sns.set()
from sklearn.metrics import roc_auc_score



SEED = 1127 

def get_size(bytes, suffix='B'):
    ''' 
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    '''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"

def get_file_size(filename):
    file_size = os.stat(filename)
    return get_size(file_size.st_size)


def get_system():
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores    :", psutil.cpu_count(logical=False))
    print("Total cores       :", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency    : {cpufreq.max:.2f} Mhz")
    print(f"Min Frequency    : {cpufreq.min:.2f} Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f} Mhz")

    print("="*40, "Memory Info", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total     : {get_size(svmem.total)}")
    print(f"Available : {get_size(svmem.available)}")
    print(f"Used      : {get_size(svmem.used)}")


    print("="*40, "Software Info", "="*40)
    print('Python     : ' + sys.version.split('\n')[0])
    print('Numpy      : ' + np.__version__)
    print('Pandas     : ' + pd.__version__)
    print('PyTorch    : ' + torch.__version__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
    if device.type == 'cuda':
        print("="*40, "GPU Info", "="*40)
        print(f'Device     : {device}')
        print(torch.cuda.get_device_name(0))
        print(f"{'Mem total': <15}: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
        print(f"{'Mem allocated': <15}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"{'Mem cached': <15}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
    
    print("="*30, "system info print done", "="*30)

def get_seed(s):
    rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

@contextmanager
def simple_timer(title):
    t0 = time()
    yield
    print("{} - done in {:.1f} seconds.\n".format(title, time() - t0))

class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    blue = "\033[94m"
    green = "\033[92m"
    yellow = "\033[93m"
    magenta = "\033[95m"
    red = "\033[91m"
    end = "\033[0m"

def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"

@contextmanager
def timer(label: str, compact=False) -> None:
    '''
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print 
    1. the time the code block takes to run
    2. the memory usage.
    '''
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    start = time()  # Setup - __enter__
    if not compact:
        print(color(f"{label}: start at {start:.2f};", color=Colors.blue))
        print(color(f"LOCAL RAM USAGE AT START: {m0:.2f} GB" , color=Colors.green))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            end = time()
            print(color(f"{label}: done at {end:.2f} ({end - start:.6f} secs elapsed);", color=Colors.blue))
            print(color(f"LOCAL RAM USAGE AT END: {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.green))
            print('\n')
    else:
        yield
        print(color(f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue))
    

def get_memory(num_var=10):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key= lambda x: -x[1])[:num_var]:
        print(color(f"{name:>30}:", color=Colors.green), 
              color(f"{get_size(size):>8}", color=Colors.magenta))

def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
    return result

def print_file_size(files):
    for file in files:
        size=get_file_size(file)
        filename = file.split('/')[-1]
        filesize = get_file_size(file)
        print(color(f"{filename:>30}:", color=Colors.green), 
              color(f"{filesize:>8}", color=Colors.magenta))

@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ", file=sys.stderr)

def get_cmap(n, cmap='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmap, n)

def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")

def roc_auc_compute_fn(y_targets, y_preds):
    '''
    roc_auc func for torch tensors
    '''
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred)

def argmax(lst):
  return lst.index(max(lst))

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        u = pickle.load(f)
    return u


if __name__ == "__main__":
    get_system()
    get_memory()