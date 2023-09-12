from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, 
                                                PrepareBaseModelConfig,
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)
import os 
from pathlib import Path

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    )-> None:
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )

        return data_ingestion_config
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_weights=self.params.WEIGHTS,
            params_in_features=self.params.IN_FEATURES,
            params_num_classes=self.params.CLASSES,
            params_pretrained=self.params.PRETRAINED,
            params_freeze_all=self.params.FREEZE_ALL,
            params_freeze_till=self.params.FREEZE_TILL,
            params_seed=self.params.SEED
        )

        return prepare_base_model_config
    


    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        # config = self.config.prepare_callbacks
        # models_ckpt_dir = os.path.dirname(config.checkpoint_models_filepath)
        
        # dirs_to_create = [
        #     Path(config.root_dir),
        #     Path(config.tensorboard_root_log_dir),
        #     Path(models_ckpt_dir),
        # ] + [Path(config.tensorboard_root_log_dir) / metric_name for metric_name in self.params.metric_names]

        # prepare_callback_config = PrepareCallbacksConfig(
        #     root_dir=Path(config.root_dir),
        #     tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
        #     checkpoint_models_filepath=Path(config.checkpoint_models_filepath)
        # )

        config = self.config.prepare_callbacks

        create_directories([Path(config.tensorboard_root_log_dir)])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            metric_names = self.params.METRIC_NAMES
        )

        return prepare_callback_config
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion
        params = self.params

        
        training_data = os.path.dirname(training.trained_best_model_filepath)
       
        create_directories([
            Path(training.root_dir),
            # Path(training_data)
        ])
       

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            model_path=Path(prepare_base_model.updated_base_model_path),
            trained_last_model_filepath=Path(training.trained_last_model_filepath),
            trained_best_model_filepath=Path(training.trained_best_model_filepath),
            results_training_model_filepath=Path(training.results_training_model_filepath),
            train_data_dir=Path(data_ingestion.unzip_dir) / 'train',
            validation_data_dir=Path(data_ingestion.unzip_dir) / 'test',
            params_epochs=params.EPOCHS,
            params_classes=params.CLASSES,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.IS_AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_device=params.DEVICE,
            params_num_workers=params.NUM_WORKERS
        )
        

        return training_config
    
        
    def get_evaluation_config(self) -> EvaluationConfig:
        
        config = self.config.evaluation
        params = self.params

        create_directories([Path(config.root_dir)])
        
        eval_config = EvaluationConfig(
            checkpoint_best_model_path=Path(config.checkpoint_best_model_path),
            data_path=Path(config.data_path),
            score_evaluation_filepath=Path(config.score_evaluation_filepath),
            params_image_size=params.IMAGE_SIZE,
            params_batch_size=params.BATCH_SIZE,
            params_classes=params.CLASSES,
            params_device=params.DEVICE,
            params_num_workers=params.NUM_WORKERS,
            params_seed = params.SEED,
            params_learning_rate = params.LEARNING_RATE
        )
        return eval_config

