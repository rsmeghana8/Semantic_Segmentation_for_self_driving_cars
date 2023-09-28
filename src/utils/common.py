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
import matplotlib.pyplot as plt
import tensorflow as tf

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

@ensure_annotations
def display(display_list : list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(unet, dataset=None, num=1):
    """
    Displays the first image of each of the num batches
    """

    for image, mask in dataset.take(num):
        pred_mask = unet.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
