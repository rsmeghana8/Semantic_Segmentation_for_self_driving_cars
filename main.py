from src.config.configuration import ConfigurationManager
from src.components.data_preparation import DataIngestion 
from src.components.data_preprocesing import DataPreprocessing
from src.components.callbacks import PrepareCallback
from src.components.model import PrepareModel
from src import logger
import tensorflow as tf



STAGE_NAME= "Training stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        prepare_data_config = config.get_data_ingestion_config()
        prepare_dataset = DataIngestion(config=prepare_data_config)
        imgs_list, masks_list = prepare_dataset.prepare_data() 

        preprocessing_config = config.get_data_processing_config()
        preprocessing = DataPreprocessing(preprocessing_config,imgs_list, masks_list)
        train, val = preprocessing.data_preprocessing()

        prepare_callbacks_config= config.get_prepare_callback_config()
        prepare_callbacks= PrepareCallback(config = prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        prepare_model_config = config.get_prepare_model_config()
        prepare_model = PrepareModel(prepare_model_config)
        model = prepare_model.unet_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        model.fit(train, validation_data=val, epochs=1, callbacks=callback_list)

if __name__ == "__main__":

    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e