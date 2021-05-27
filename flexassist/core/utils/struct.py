from   functools import reduce
import itertools
import os 


def flatten_list(lst:list):
    "flatten a list or string into a single list"
    for x in lst:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flatten_list(x):
                yield y
        else:
            yield x

def flatten_paths(data:dict) -> dict :
    "flatten all elements within a given dictionary, joining filesystem paths"
    return {k: v if isinstance(v,str)
               else os.path.join(* list(flatten_list(v)) ) 
            for k, v in data.items()}
        

