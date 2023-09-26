from dataclasses import dataclass 
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    Buffer_size: int
    Batch_size: int

@dataclass(frozen=True)
class PrepareCallbackConfig:
    root_dir: Path
    tensorboard_root_log_dir:Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    input_size: tuple
    n_classes: int
 