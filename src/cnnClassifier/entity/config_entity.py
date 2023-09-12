from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_weights: str
    params_in_features: int
    params_num_classes: int
    params_pretrained: bool
    params_freeze_all: bool 
    params_freeze_till: int 
    params_seed: int



@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    metric_names: list



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    model_path: Path
    trained_last_model_filepath: Path
    trained_best_model_filepath: Path
    results_training_model_filepath: Path
    train_data_dir: Path
    validation_data_dir: Path
    params_epochs: int
    params_classes: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_device: str
    params_num_workers: int



@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_best_model_path: Path
    data_path: Path
    score_evaluation_filepath: Path
    params_image_size: list
    params_batch_size: int
    params_classes: int
    params_device: str
    params_num_workers: int
    params_seed: int
    params_learning_rate: float