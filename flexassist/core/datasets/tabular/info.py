import  numpy  as np 
import  pandas as pd 
from    typing import List 

# framewise statistics
get_uniques         = lambda df: df.nunique(axis=0)

class InfoTabular(object):
    def __init__(self, df:pd.DataFrame):
        super().__init__()
        # get missing information
        n_rows, n_cols  = df.shape
        df_stats        = self.get_missing_stats(df)
        missing_pct     = round( df_stats['count'].sum() / (n_rows * n_cols), 4) * 100

    def describe(self, df:pd.DataFrame, k:int=None):
        "describe the dataframe properties"
        print(f"DataFrame Shape: {df.shape}")
        print(f"Columns:         {list(df.columns[:k]) if k else list(df.columns)}")

    def describe_dt(self, ds:pd.Series, name:str):
        """describe a time series partition
        """
        dt_day_span     = lambda xs:  int((ds.max() - ds.min()).days)
        dt_year_span    = lambda xs:  round( dt_day_span(xs) / 365, 3)
        dt_min          = ds.min()
        dt_max          = ds.max()
        return pd.DataFrame.from_dict( dict(min=dt_min, 
                                            max=dt_max,
                                            num_days=dt_day_span(ds), 
                                            num_years=dt_year_span(ds)), orient='index', columns=[name])

    def describe_col_continuous(self, ds: pd.Series) -> dict:
        "statistically describe the dataframe"
        ds_custom_metrics =  pd.Series({
            'skewness': ds.skew(), 
            'kurtosis': ds.kurt()
        }).round(3)
        return pd.DataFrame(pd.concat([ds.describe(), ds_custom_metrics]).round(3), columns=[ds.name]).T

    def describe_col_categorical(self, part:List[pd.DataFrame], indices:List[str], col:str='target') -> pd.DataFrame:
        """
        Examples: 
        >>> describe_col_categorical(part=[train, val, test], indices=['train', 'val', 'test'])
        """
        return pd.DataFrame([data[col].value_counts(sort=False)  for data in  part], index=indices)

    def get_missing_stats(self, df:pd.DataFrame) -> pd.DataFrame:
        """get count and percentage missing for each columnar data
        sns.heatmap(df_cc.isnull(), yticklabels = False, cbar = False, cmap="Blues")

        Examples:
        # Example 1: across columns
        >>> missing_props = df.isna().sum() / len(df)
        >>> missing_props[missing_props > 0].sort_values(ascending=False)
        # Example 2: for a single column
        >>> df[col].value_counts(dropna=False, normalize=True).head()
        """
        # cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
        # num_null = df_feat.loc[df_feat.isnull().any(axis=1), ].shape[0]
        return pd.DataFrame(zip(df.isnull().sum(), df.isnull().sum()/len(df)), 
                            columns=['count', 'pct'], 
                            index=df.columns).sort_values(by=['pct'], ascending=False)





class InfoTabularPartition(object):
    def describe_partition(self, train:tuple, validation:tuple, test:tuple=None) -> pd.DataFrame:
        """describe partition sizes"""
        inp_train, tgt_train = train
        inp_val,   tgt_val   = validation
        inp_test,  tgt_test  = test if test else ([], [])
        cols   = ["Inp-Train", "Tgt-Train", "Inp-Validation", "Tgt-Validation", "Inp-Test", "Tgt-Test"]
        sizes  = [len(inp_train), len(tgt_train), len(inp_val), len(tgt_val), len(inp_test), len(tgt_test)]
        df_partition = pd.DataFrame(sizes, index=cols).rename(columns={0:'size'}).T
        return df_partition
