import  glob 
import  os 
import  numpy  as np 
import  modin.pandas as pd          #import  pandas as pd 
from    typing import List
from    pathlib import Path 

import  flexassist.core.system.reader as rd
from    flexassist.core.utils.struct import flatten_paths

def load_config(config_path:str) -> dict:
    "load set of yaml configurations"
    defaults_config:dict   = rd.read_yaml(Path(config_path)/'defaults.yaml')
    infra_config:dict      = rd.read_yaml(Path(config_path)/'env.yaml')
    dataset_config:dict    = rd.read_yaml(Path(config_path)/'dataset.yaml')
    training_config:dict   = rd.read_yaml(Path(config_path)/'training.yaml')
    modelmeta_config:dict  = rd.read_yaml(Path(config_path)/'modelmeta.yaml')
    # for when we have nested
    experiment_config:dict = rd.read_yaml(Path(config_path)/'experiment.yaml')
    experiment_config:dict = dict(experiment_parameters=flatten_paths(experiment_config['paths']))
    # build a final config for context usage
    config = {
        **defaults_config, 
        **dataset_config, 
        **training_config,
        **modelmeta_config,
        **infra_config, 
        **experiment_config
    }
    return config 

