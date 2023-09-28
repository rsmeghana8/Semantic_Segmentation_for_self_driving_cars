from src.constants import *
from src.utils.common import read_yaml,create_directories
from src.entity.config_entity import DataIngestionConfig, PrepareCallbackConfig, PrepareModelConfig, DataPreprocessingConfig


class ConfigurationManager:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            params_filepath= PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)-> DataIngestionConfig:
            config = self.config.data_ingestion
            
            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir= config.root_dir )

            return data_ingestion_config
    
    def get_data_processing_config(self) -> DataPreprocessingConfig:
         
         config = self.params

         prepare_processing_config = DataPreprocessingConfig(
              Buffer_size = config.BUFFER_SIZE,
              Batch_size = config.BATCH_SIZE
         )
         return prepare_processing_config
    
    def get_prepare_callback_config(self)-> PrepareCallbackConfig:
        config = self.config.prepare_callbacks

        create_directories([config.root_dir])


        create_directories([ 
                            Path(config.checkpoint_model_filepath),
                            Path(config.tensorboard_root_log_dir)  ])

        prepare_callback_config = PrepareCallbackConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath= Path(config.checkpoint_model_filepath)
    
        )

        return prepare_callback_config
    
    def get_prepare_model_config(self)-> PrepareModelConfig:
        config = self.params



        prepare_model_config = PrepareModelConfig(
                            input_size = config.input_size,
                            n_classes= config.n_classes,
                            )
        return prepare_model_config
    