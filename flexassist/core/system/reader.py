import yaml 
import json 


def read_yaml(filename:str) -> dict:
    "read context of a yaml file"
    with open(filename) as f: 
        data:dict = yaml.load(f, Loader=yaml.FullLoader)
    return data
   
def read_json(filename:str) -> str:
    "read a json string from a file"
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

