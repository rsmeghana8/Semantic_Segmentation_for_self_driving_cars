import os
from src import logger
from random import choice
import shutil
from src.utils.common import read_yaml,create_directories
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
import glob

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config=config

    def prepare_data(self):
        root_dir = self.config.root_dir
        imgs_list_1=[]
        masks_list_1=[]

        temp_a = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            temp_a.append(dirnames)
            data_dirs = sorted(temp_a[0])[:5]

        for folder in data_dirs:
            imgs_list_1.append(glob.glob(root_dir +'/' + folder+'/'+ folder+'/CameraRGB/*.png'))
            masks_list_1.append(glob.glob(root_dir +'/' + folder+'/'+ folder+'/CameraSeg/*.png'))


        # Transforming the list of lists of images and masks into one list for images and one for masks
        imgs_list = []
        masks_list = []

        for sublist in imgs_list_1:
            imgs_list.extend(sublist)
        for sublist in masks_list_1:
            masks_list.extend(sublist)

        return imgs_list,masks_list