from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.components.metrics import Metrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.efficientnet import EfficientNet
from pathlib import Path
from box import ConfigBox
from cnnClassifier import logger
import torch
import os
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch import nn
from torch import optim  
import pandas as pd
from torchvision import transforms
from typing import Dict



class Training:
    def __init__(self, config: TrainingConfig):
        self.root_dir = config.root_dir
        self.model_path = config.model_path
        self.trained_last_model_filepath = config.trained_last_model_filepath
        self.trained_best_model_filepath = config.trained_best_model_filepath
        self.results_training_model_filepath = config.results_training_model_filepath
        self.train_data_dir = config.train_data_dir
        self.validation_data_dir = config.validation_data_dir
        self.params_epochs = config.params_epochs
        self.params_classes = config.params_classes
        self.params_batch_size = config.params_batch_size
        self.params_is_augmentation = config.params_is_augmentation 
        self.params_image_size = config.params_image_size
        self.params_learning_rate = config.params_learning_rate
        self.params_device = "cuda" if config.params_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.params_num_workers = min(config.params_num_workers, os.cpu_count())

        
        self.data_transforms =  self._data_transforms() if self.params_is_augmentation else ConfigBox({'data_train_transform': None, 'data_val_transform':None})
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.dataloader = None
        self.epsilon = 1e-7
    
    def get_base_model(self):
        logger.info("Getting the base model...")

        try:
            # Load the model from the specified path and move it to the appropriate device
            self.model = torch.load(self.model_path).to(self.params_device)

            # Check if the model type is as expected (EfficientNet)
            if not isinstance(self.model, EfficientNet):
                assertion_msg = f"The model must be of type EfficientNet, not {type(self.model)} or any other type"
                raise AssertionError(assertion_msg)

        except AssertionError as ae:
            # Log the assertion error and its message
            logger.error("Assertion error: %s", str(ae))
        except Exception as e:
            # In case of any other error, log the exception
            logger.exception("An error occurred while loading the base model: %s", str(e))
        
        logger.info("Base model loaded successfully.")
        # logger.info("Base model retrieval completed.")

    
    
    def _data_transforms(self):
        data_train_transform = transforms.Compose([
            transforms.Resize(tuple(self.params_image_size[1:])),
            # transforms.RandomHorizontalFlip(),  # Random horizontal flip
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Perturb colors
            # transforms.RandomRotation(10),  # Random rotation by ±10 degrees
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Log the transformations
        logger.info("Data Train Transformations:\n%s", data_train_transform)


        data_val_transform = transforms.Compose([
            transforms.Resize(tuple(self.params_image_size[1:])),
            # transforms.RandomRotation(10),  # Random rotation by ±10 degrees
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Log the transformations
        logger.info("Data Validation Transformations:\n%s", data_val_transform)

        data_transforms = ConfigBox(
                            {
                                'data_train_transform': data_train_transform,
                                'data_val_transform': data_val_transform
                            }
                        )
            
        
        return data_transforms


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

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(self.train_data_dir, transform=self.data_transforms.data_train_transform)
        val_data = datasets.ImageFolder(self.validation_data_dir, transform=self.data_transforms.data_val_transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.params_batch_size,
            shuffle=True,
            num_workers=self.params_num_workers,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=self.params_batch_size,
            shuffle=True,
            num_workers=self.params_num_workers,
            pin_memory=True,
        )

        self.dataloader = ConfigBox(
                            
                                {
                                    'train_dataloader': train_dataloader, 
                                    'val_dataloader': val_dataloader, 
                                    'class_names': class_names
                                }
                            
                        )
        
        logger.info("Dataloader created successfully.")

    

    def set_loss_optimizer(self):
        logger.info(f"Setting loss function: CrossEntropyLoss")
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info(f"Setting optimizer: Adam with learning rate {self.params_learning_rate}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params_learning_rate)




    def train_step(self):
        """Trains a PyTorch model for a single epoch with additional metrics.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            model: A PyTorch model to be trained.
            dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
            A tuple of training loss and training metrics.
            In the form (train_loss, train_metrics). For example:

            (0.1112, {'accuracy': 0.8743, 'recall': 0.8501, 'f1_score': 0.8952, 'mAP50': 0.9123, 'mAP90': 0.7890})
        """
        
        # Put model in train mode
        self.model.train()

        # Setup train loss and train metric values
        loss, accuracy, precision, recall, f1_score, mAP50, mAP90 = 0, 0, 0, 0, 0, 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(self.dataloader.train_dataloader):
            # Send data to target device
            X, y = X.to(self.params_device), y.to(self.params_device)

            # 1. Forward pass
            y_pred = self.model(X)

            # 2. Calculate and accumulate loss
            loss = self.loss_fn(y_pred, y)
            loss += loss.item() 

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

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
        num_batches = len(self.dataloader.train_dataloader)
        loss /= num_batches
        accuracy /= num_batches
        precision /= num_batches
        recall /= num_batches
        f1_score /= num_batches
        mAP50 /= num_batches
        mAP90 /= num_batches

        # Return a dictionary with all the metrics
        train_metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP50': mAP50,
            'mAP90': mAP90
        }

        return ConfigBox(train_metrics)



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

        # Put model in evaluation mode
        self.model.eval()

        # Setup train loss and train metric values
        loss, accuracy, precision, recall, f1_score, mAP50, mAP90 = 0, 0, 0, 0, 0, 0, 0

        # Disable gradient computation during validation
        with torch.no_grad():
            for batch, (X, y) in enumerate(self.dataloader.val_dataloader):
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
        num_batches = len(self.dataloader.val_dataloader)
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

        return ConfigBox(val_metrics)


    def train(self, tb_writer=None):
        """
        Trains and tests a PyTorch model.

        Args:
            tb_writer: TensorBoard writer for logging (optional).

        Returns:
            A dictionary of training and testing metrics.
        """
        metrics = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_recall": [],
            "val_recall": [],
            "train_f1_score": [],
            "val_f1_score": [],
            "train_mAP50": [],
            "val_mAP50": [],
            "train_mAP90": [],
            "val_mAP90": []
        }
        best_val_acc = 0.0  # Initialize best validation accuracy

        for epoch in tqdm(range(self.params_epochs)):
            train_metrics = self.train_step()
            val_metrics = self.validate_step()

            # Log progress
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "val_loss": val_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_accuracy": val_metrics.accuracy,
                "train_precision": train_metrics.precision,
                "val_precision": val_metrics.precision,
                "train_recall": train_metrics.recall,
                "val_recall": val_metrics.recall,
                "train_f1_score": train_metrics.f1_score,
                "val_f1_score": val_metrics.f1_score,
                "train_mAP50": train_metrics.mAP50,
                "val_mAP50": val_metrics.mAP50,
                "train_mAP90": train_metrics.mAP90,
                "val_mAP90": val_metrics.mAP90
            }
            metrics.update(epoch_metrics)
            self.log_metrics(epoch_metrics)

            # Save metrics
            # self.save_metrics(metrics)

            # Update best_val_acc if current val_acc is greater
            best_val_acc = max(best_val_acc, val_metrics.accuracy)

            # Save last trained model
            self.__class__.save_model(self.trained_last_model_filepath, self.model, self.optimizer, self.data_transforms, self.loss_fn, metrics)

            # Save best trained model if validation accuracy improved
            if val_metrics.accuracy >= best_val_acc or epoch == 0:
                self.__class__.save_model(self.trained_best_model_filepath, self.model, self.optimizer, self.loss_fn, self.data_transforms, metrics)

            # Log metrics to TensorBoard if writer is provided
            if tb_writer is not None:
                tb_writer.log_metrics(ConfigBox(metrics))

        if tb_writer is not None:
            # Close TensorBoard writer if provided
            tb_writer.close_writer()



        
    def log_metrics(slef, metrics):
        """Logs epoch metrics."""
        log_str = " | ".join([f"{key}: {value:.4f}" for key, value in metrics.items() if key != 'epoch'])
        logger.info(f"Epoch {metrics['epoch']} - {log_str}")


    def save_metrics(self,  metrics):
        """Saves metrics to a CSV file."""
        pd.DataFrame(metrics).to_csv(self.results_training_model_filepath, index=False)


    @staticmethod
    def save_model( path: Path, 
                    model: EfficientNet, 
                    optimizer: torch.optim, 
                    loss_fn: nn.CrossEntropyLoss,
                    data_transforms: ConfigBox,
                    metrics: ConfigBox) -> None:
        """
        Saves the model's state dictionary to a specified path.

        Parameters:
        - path (str): The file path to save the model.
        - model: The model to be saved.

        No return value.
        """
        logger.info(f"Saving model to {path}...")
        # Save the model's state dictionary to the specified path
        torch.save(model.state_dict(), path)
        torch.save({
                    **{
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'data_transforms': data_transforms,
                        'loss_fn': loss_fn
                    }, **metrics
                }, path)
        
        logger.info("Model saved.")