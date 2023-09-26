import tensorflow as tf
from src.entity.config_entity import DataPreprocessingConfig
import numpy

class DataPreprocessing:
    def __init__(self, config : DataPreprocessingConfig,  img_list: list, mask_list: list):
        self.img_list = img_list
        self.mask_list = mask_list
        self.config = config

    @staticmethod
    def process_path(img_path, mask_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img= tf.image.convert_image_dtype(img,tf.float32)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask,axis =-1, keepdims=True)

        return img, mask

    @staticmethod
    def preprocess(image, mask):
        input_image=tf.image.resize(image,(192,256), method='nearest')
        input_mask=tf.image.resize(mask,(192,256),method='nearest')

        return input_image, input_mask



    def data_preprocessing(self):

        #Combining images and masks into pairs
        image_filenames = tf.constant(self.img_list)
        mask_filenames = tf.constant(self.mask_list)
        dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))

            

        image_ds = dataset.map(self.process_path)
        processed_image_ds = image_ds.map(self.preprocess) 

        map_dataset = processed_image_ds.cache().shuffle(self.config.Buffer_size).batch(self.config.Batch_size)
        num_samples = map_dataset.reduce(0, lambda x, _: x + 1).numpy()

        train_dataset = map_dataset.take(int(0.8 * num_samples))
        val_dataset = map_dataset.skip(int(0.8 * num_samples))

        return train_dataset , val_dataset
    

         
    

         