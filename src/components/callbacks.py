from src.entity.config_entity import PrepareCallbackConfig
import time
import os
import tensorflow as tf

class PrepareCallback:
    def __init__(self, config : PrepareCallbackConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(self.config.tensorboard_root_log_dir, f"tb_logs_at_{timestamp}",)
        print(tb_running_log_dir)
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    
    @property
    def _create_ckpt_callbacks(self):
        
        return tf.keras.callbacks.ModelCheckpoint(
            filepath= os.path.join(self.config.checkpoint_model_filepath,"weights-{epoch:03d}-{val_loss:.4f}.hdf5"),
            save_best_only=True)
    @property
    def _create_reduce_lr(self):

        return tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=1, min_lr=1e-6)
    @property
    def _create_early_stop(self):
             
        return tf.keras.callbacks.EarlyStopping(patience=6)
    
    def get_tb_ckpt_callbacks(self):
        return [self._create_tb_callbacks, self._create_ckpt_callbacks,self._create_reduce_lr, self._create_early_stop] 
