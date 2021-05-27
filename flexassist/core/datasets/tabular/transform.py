import  numpy as np 
import  pandas as pd 
from    typing import List




def trsfrm_col_categorical(df:pd.DataFrame, cols:List[str]):
    "convert col as categorical rather than object"
    df_trsfrm = df.copy()
    for col in cols:
        df_trsfrm[col] = df_trsfrm[col].astype('category')
    return df_trsfrm 


def trsfrm_aggregeate_mulindex(df:pd.DataFrame, 
                               grouped_cols:List[str], 
                               agg_col:str, 
                               operation:str, 
                               k:int=5):
    """transform aggregate statistics for multiindex
    
    Examples:     
    >>> df_agg = trsfrm_aggregeate_mulindex( df_train, ["store", "item"], 'sales', 'mean')
    """
    cols           = ["sum", "mean", "median", "std", "min", "max", "skew"]
    lvl0,lvl1      = grouped_cols
    df_agg         = pd.DataFrame( df.groupby(grouped_cols)[agg_col].agg(cols) )[operation]
    df_agg         = df_agg.groupby(level=lvl0).nlargest(k).reset_index(level=1, drop=True)
    df_agg         = df_agg.reset_index()
    df_agg[lvl1]   = df_agg.item.astype('category')
    return df_agg
