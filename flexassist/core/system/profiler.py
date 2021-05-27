import numpy  as np 
import pandas as pd
import os  
def get_memory_usage():
    return np.round(os.psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
 
def reduce_mem_usage(df:pd.DataFrame, verbose=True):
    start_mem         = df.memory_usage().sum() / 1024**2    
    float_cols        = [c for c in df if df[c].dtype == 'float64']
    int_cols          = [c for c in df if df[c].dtype in ['int64', 'int32']]
    df[float_cols]    = df[ float_cols].astype(np.float16)
    df[int_cols]      = df[int_cols].astype(np.int16)
    end_mem           = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
