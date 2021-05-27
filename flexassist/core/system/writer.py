import numpy   as np 
import pandas  as pd 
from   pathlib import Path 

def write_csv(df: pd.DataFrame, path:str, name:str, with_index:bool=False) -> str:
    "write dataframe csv file to path with given filename"
    file_path = Path(path)/name
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=with_index)
    return str(file_path) 
