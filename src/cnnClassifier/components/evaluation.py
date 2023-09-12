from torchvision import datasets, transforms
from pathlib import Path
import os
from box import ConfigBox
import torch
from torch.utils.data import DataLoader
from cnnClassifier.components.metrics import Metrics 
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from cnnClassifier import logger
from tqdm.auto import tqdm
from torch import nn
from torch import optim  
import torchvision
import numpy as np





class Evaluation:
    def __init__(self, config: EvaluationConfig):
        # Get configuration params
        self.config = config
        self.data_path = config.data_path
        self.checkpoint_best_model_path = config.checkpoint_best_model_path
        self.score_evaluation_filepath = config.score_evaluation_filepath
        self.params_image_size = config.params_image_size
        self.params_batch_size = config.params_batch_size
        self.params_classes = config.params_classes
        self.params_seed = config.params_seed
        self.params_learning_rate = config.params_learning_rate
        self.params_device = "cuda" if config.params_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.params_num_workers = min(config.params_num_workers, os.cpu_count())

        # Get the checkpoint
        self.checkpoint = self.load_checkpoint(self.checkpoint_best_model_path, self.params_learning_rate)
        self.model = self.checkpoint.model
        self.loss_fn = self.checkpoint.loss_fn
        self.optimizer = self.checkpoint.optimizer
        self.data_transforms = self.checkpoint.data_transforms
        self.dataloader = self.create_dataloader()
        self.epsilon = 1e-7
    

    def create_dataloader(self):
        """Creates training and testing DataLoader.

        Takes in a training directory and testing directory path and turns
        them into PyTorch Datasets and then into PyTorch DataLoader.

        Args:
            train_dir: Path to training directory.
            val_dir: Path to testing directory.
            transform: torchvision transforms to perform on training and testing data.
            batch_size: Number of samples per batch in each of the DataLoader.
            num_workers: An integer for number of workers per DataLoader.

        Returns:
            A tuple of (train_dataloader, val_dataloader, class_names).
            Where class_names is a list of the target classes.
            Example usage:
            train_dataloader, val_dataloader, class_names = \
                = create_dataloader(train_dir=path/to/train_dir,
                                    val_dir=path/to/val_dir,
                                    transform=some_transform,
                                    batch_size=32,
                                    num_workers=4)
        """
        logger.info("Creating training and validation dataloader...")

        data = datasets.ImageFolder(self.data_path, transform=self.data_transforms)

        # Get class names
        class_names = data.classes

        dataloader = DataLoader(
            data,
            batch_size=self.params_batch_size,
            shuffle=True,
            num_workers=self.params_num_workers,
            pin_memory=True,
        )
    
        logger.info("Dataloader created successfully.")


        return dataloader 
                                    

        
    
    def validate_step(self):
        """Validates a PyTorch model for a single epoch with additional metrics.

        Turns a target PyTorch model to evaluation mode and then
        runs through all of the required validation steps (forward
        pass, loss calculation, and metric calculation).

        Args:
            model: A PyTorch model to be validated.
            dataloader: A DataLoader instance for the model to be validated on.
            loss_fn: A PyTorch loss function for validation loss calculation.
            device: A target device to compute on (e.g., "cuda" or "cpu").

        Returns:
            A dictionary of validation metrics.
            For example:

            {'loss': 0.1234, 'accuracy': 0.8901, 'recall': 0.8501, 'f1_score': 0.8952, 'mAP50': 0.9123, 'mAP90': 0.7890}
        """
        logger.info('Validation started...')
        
        # Put model in evaluation mode
        self.model.eval()

        # Setup train loss and train metric values
        loss, accuracy, precision, recall, f1_score, mAP50, mAP90 = 0, 0, 0, 0, 0, 0, 0

        # Disable gradient computation during validation
        with torch.no_grad():
            for batch, (X, y) in tqdm(enumerate(self.dataloader)):
                # Send data to target device
                X, y = X.to(self.params_device), y.to(self.params_device)

                # 1. Forward pass
                y_pred = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(y_pred, y)
                loss += loss.item()

                # Calculate and accumulate accuracy metric across all batches
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1).int()

                # Calculate and accumulate recall, F1 score, mAP50, and mAP90
                metrics = Metrics.calculate_metrics_macro(y_pred_class, y, self.params_classes)
                accuracy += metrics.accuracy
                precision += metrics.precision
                recall += metrics.recall
                f1_score += metrics.f1_score
                mAP50 += metrics.mAP50 
                mAP90 += metrics.mAP90

        # Adjust metrics to get average loss and metrics per batch 
        num_batches = len(self.dataloader)
        loss /= num_batches
        accuracy /= num_batches
        precision /= num_batches
        recall /= num_batches
        f1_score /= num_batches
        mAP50 /= num_batches
        mAP90 /= num_batches

        # Return a dictionary with all the metrics
        val_metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP50': mAP50,
            'mAP90': mAP90
        }

        logger.info("Validation completed successfully.")

        return val_metrics
    
    @staticmethod
    def load_checkpoint(path: Path, params_learning_rate):
        logger.info(f"Loading model from {path}...")
        # Get model
        torch.manual_seed(122)
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=3)
        )

        # Get optimozer
        optimizer = optim.Adam(model.parameters(), lr=params_learning_rate) 

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        data_transforms = checkpoint['data_transforms']
        loss_fn = checkpoint['loss_fn']

        logger.info(f"Model loaded successfully.")

        # Créez un dictionnaire avec les données chargées et retournez-le
        loaded_data = {
            'model': model,
            'optimizer': optimizer,
            'data_transforms': data_transforms.data_val_transform,
            'loss_fn': loss_fn
        }
        

        return ConfigBox(loaded_data)


    

    def evaluation(self):
        metrics = self.validate_step()
        self.save_score(metrics, self.score_evaluation_filepath)

    @staticmethod
    def save_score(metrics, path_to_save):
        logger.info(f'Saving metrics to {path_to_save}...')
        for key in metrics.keys():
            metrics[key] = metrics[key].cpu().numpy().tolist() if isinstance(metrics[key], torch.Tensor) else metrics[key]
        save_json(path=path_to_save, data=metrics)
        logger.info(f'Metrics Saved successfully.')



    