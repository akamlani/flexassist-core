import numpy   as np 
import pandas  as pd 
import json 
import pickle 
import shutil 
from   typing  import List 
from   pathlib import Path 

copy_to_dest = lambda src_dir, dst_dir: shutil.copy(src_dir, dst_dir)


def create_directories(data_dir="data", project_dir="mnist"):
    "create directory structure"
    data_path  = Path(data_dir)
    path       = data_path / project_dir
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(data_dict:dict, path:str, name:str) -> None:
    """dump contents to a file via json"""
    file_path = Path(path)/name
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def write_records_json(records:List[dict], path:str, name:str):
    """
    records: records should be a list of dictionaries
    """
    file_path = Path(path)/name
    with open(file_path, "w") as f:
        f.write(json.dumps(records))

def write_record(record:dict, path:str, name:str):
    """ write credentials to a file

    Arguments:
        path (str): path to where json credentials file is located
        creds (dict): dictionary of username, password credentials
    
    Example:
        >>> path  = Path('~/.kaggle/kaggle.json')
        >>> creds = '{"username":"xxx","key":"xxx"}'
        >>> write_creds(path, creds)
    """
    record    = json.dumps(record) 
    file_path = (Path(path)/name).expanduser()
    if not file_path.exists():
        file_path.parent.mkdir(exist_ok=True)
        file_path.write(record)
        file_path.chmod(0o600)

def write_csv(df: pd.DataFrame, path:str, name:str, with_index:bool=False) -> str:
    "write dataframe csv file to path with given filename"
    file_path = Path(path)/name
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=with_index)
    return str(file_path) 

def write_pickle(data, path:str, name:str):
    """write binary contents to a pickle file"""
    file_path = Path(path)/name
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

