import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from src import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """reads yaml file
    
    Args:path_to_yaml (str): path to yaml file
    
    Raises: 
     ValueError: if yaml file is empty
     e: empty file
     
    Returns:

     ConfigBox: configuration box 
     """
    try: 
        with open(path_to_yaml) as yaml_file:
            content =yaml.safe_load(yaml_file)
            logger.info(f"yaml file{path_to_yaml}loaded Successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError(f"yaml file{path_to_yaml} is empty")
    except Exception as e:
        raise e    

@ensure_annotations
def create_directories(path_to_directories: list , verbose=True):
    """creates directories
    
    Args: 
        path_to_directories(list): list of path of directories
        
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'created directory at:{path}')