import numpy  as np 
import pandas as pd 
import math 
from   typing import List 
from   pandas.tseries.holiday import USFederalHolidayCalendar as calendar



# constants
MINUTES_IN_HOUR = 60
HOURS_IN_DAY    = 24

# datetime statistics
trsfrm_to_dt    = lambda ds: pd.to_datetime(ds, errors = 'coerce')
dt_day_span     = lambda xs:  int((xs.max() - xs.min()).days)
dt_year_span    = lambda xs:  round( dt_day_span(xs) / 365, 3)

def get_temporal_duration(df, cols:List[str], prefix:str):
    "calculate the duration between two separate columns"
    col1, col2 = cols
    df[f"{prefix}_duration"]     = df[col1] - df[col2]
    df[f"{prefix}_duration_min"] = df[f"{prefix}_duration"].dt.seconds/60
    return df

def get_lifespans(date_span:dict) -> pd.DataFrame:
    "calculate day and year lifespans per a dictionary of durations (days)"
    df_co_dt_span = pd.DataFrame.from_dict(date_span, orient='index', columns=['days'])
    df_co_dt_span['years'] = round( df_co_dt_span['days']/365, 3)
    return df_co_dt_span.sort_values(by='days', ascending=False)




def trsfrm_temporal_datetime(df:pd.DataFrame, col:str, prefix:str='dt') -> pd.DataFrame:
    """create time-based features, to be created on each individual partition
    
    Example:
    >>> df_train = tstrsfrm.trsfrm_temporal_datetime(df_train, prefix='dt')
    """
    # expecting 'date' like field column here
    df_dt = df.copy()
    # df.date.dt.__getattribute__(attr_name)
    df_dt[f'{prefix}:date']             = pd.to_datetime(df_dt[col].dt.date)
    # days across the entire year
    df_dt[f'{prefix}:doy']              = df_dt[col].dt.dayofyear
    # dt.weekday: 0(Mon)-6(Sunday)
    df_dt[f'{prefix}:day']              = df_dt[col].dt.day         
    df_dt[f'{prefix}:dow']              = df_dt[col].dt.dayofweek    
    df_dt[f'{prefix}:dow_name']         = df_dt[col].dt.day_name().transform(lambda xs: xs.lower())
    # basic components
    df_dt[f'{prefix}:hour']             = df_dt[col].dt.hour
    df_dt[f'{prefix}:minute']           = df_dt[col].dt.minute
    df_dt[f'{prefix}:month']            = df_dt[col].dt.month
    df_dt[f'{prefix}:year']             = df_dt[col].dt.year
    # day part: 
    df_dt[f'{prefix}:day_part']         = df_dt[f'{prefix}:hour'].apply(day_part)
    # boolean flags 
    start_date, end_date                = (df_dt[f'{prefix}:date'].min(), df_dt[f'{prefix}:date'].max())
    holidays                            = calendar().holidays(start=start_date, end=end_date)
    df_dt[f'{prefix}:is_holiday']       = df_dt[f'{prefix}:date'].isin(holidays).astype(int)

    df_dt[f'{prefix}:is_weekend']       = np.where(df_dt[f'{prefix}:dow'].isin([5,6]), 1,0)
    df_dt[f'{prefix}:is_year_start']    = df_dt[col].dt.is_year_start.astype(int)
    df_dt[f'{prefix}:is_quarter_start'] = df_dt[col].dt.is_quarter_start.astype(int)
    df_dt[f'{prefix}:is_quarter_end']   = df_dt[col].dt.is_quarter_end.astype(int)
    df_dt[f'{prefix}:is_month_start']   = df_dt[col].dt.is_month_start.astype(int)
    df_dt[f'{prefix}:is_month_end']     = df_dt[col].dt.is_month_end.astype(int)

    # fractional transforms can be cyclic if available
    # fractional hour in range 0-24, e.g. 12h30m --> 12.5: accurate to 1 minute.
    df_dt[f'{prefix}:fractionalhour']   = df_dt[f'{prefix}:hour']  + df_dt[f'{prefix}:minute'] / MINUTES_IN_HOUR
    # fractional day in range 0-1, e.g. 12h30m --> 0.521:  accurate to 1 minute
    df_dt[f'{prefix}:fractionalday']    = df_dt[f'{prefix}:fractionalhour'] / HOURS_IN_DAY
    # fractional in months: get_frac_months = lambda xs: xs / np.timedelta64(1, 'M')
    return df_dt 

def day_part(hour):
    "assign literals for specific hours"
    if hour in [4,5]:
        return "dawn"
    elif hour in [6,7]:
        return "early morning"
    elif hour in [8,9,10]:
        return "late morning"
    elif hour in [11,12,13]:
        return "noon"
    elif hour in [14,15,16]:
        return "afternoon"
    elif hour in [17, 18,19]:
        return "evening"
    elif hour in [20, 21, 22]:
        return "night"
    elif hour in [23,24,1,2,3]:
        return "midnight"

def trsfrm_cyclic(df:pd.Series, col:str, period_type:str) -> pd.DataFrame:
    """transform cyclic periodic features
    
    Examples:
    periodic_pipe = make_pipeline(
        FunctionTransformer( trsfrm_cyclic, kw_args={'col': 'dt_doy',   'period_type': 'day'},     validate=False ),
        FunctionTransformer( trsfrm_cyclic, kw_args={'col': 'dt_dow',   'period_type': 'week'},    validate=False ),
        FunctionTransformer( trsfrm_cyclic, kw_args={'col': 'dt_month', 'period_type': 'month'},   validate=False )
    )
    X  = df_train[['dt_doy', 'dt_dow', 'dt_month']].copy()
    Xp = periodic_pipe.fit_transform(X)
    """
    periods = dict(
        minute = 60,    # 60 minutes in an hour
        hour   = 24,    # 24 hours in a day
        day    = 365,   # 365 days in year
        week   = 7,     # 7 days in a year
        month  = 12     # 12 months in a year
    )
    
    df_cyclic = df.copy() 
    # divide by ( period - 1 ) if values are zero(0) indexed
    get_cyclic                      = lambda xs, period: 2 * math.pi * xs / (period - 1)
    df_cyclic.loc[:, f'{col}_norm'] = get_cyclic(df[col], periods[period_type])
    df_cyclic.loc[:, f'{col}_cos']  = np.cos(df_cyclic[f'{col}_norm'])
    df_cyclic.loc[:, f'{col}_sin']  = np.sin(df_cyclic[f'{col}_norm'])
    return df_cyclic 