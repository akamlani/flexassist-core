import numpy  as np 
import pandas as pd 
import glob 
from   pathlib import Path
from   typing  import List 
from   utils   import config as cfg 

from   flexassist.core.system.profiler import reduce_mem_usage
from   flexassist.core.system.writer   import write_csv
from   flexassist.core.system.reader   import read_yaml 
from   flexassist.core.utils.struct    import flatten_paths


# from top level directory: run as python -m programs.transform_inputs
if __name__ == "__main__":
    np.random.seed(42)
    config_files:List[str] = glob.glob('./config' + '/*')
    config = cfg.load_config('./config')

    data_proj_dir  = config['dataset']['path']
    df_cal         = pd.read_csv(Path(data_proj_dir)/"calendar.csv", parse_dates=['date'])
    df_prices      = pd.read_csv(Path(data_proj_dir)/"sell_prices.csv")
    df_train       = pd.read_csv(Path(data_proj_dir)/"sales_train_evaluation.csv")          # valiation: d_1 - d_1913, evaluation: d_1 - d_1941
    df_train['id'] = df_train.id.str.replace('_evaluation', '')
    # reduce memory usage for later use 
    df_cal         = reduce_mem_usage(df_cal)
    df_prices      = reduce_mem_usage(df_prices)
    df_train       = reduce_mem_usage(df_train)

    # save reduced memory to disk
    datasets_dir   = config['experiment_parameters']['datasets_dir']
    write_csv(df_cal,    Path(datasets_dir)/'transformers', 'calendar.csv')    
    write_csv(df_prices, Path(datasets_dir)/'transformers', 'sell_prices.csv')    
    write_csv(df_train,  Path(datasets_dir)/'transformers', 'train.csv')    