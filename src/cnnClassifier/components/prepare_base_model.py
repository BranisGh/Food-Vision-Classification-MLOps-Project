import torch
import torch.nn as nn
import torchvision
from torchvision.models.efficientnet import EfficientNet
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier import logger
from pathlib import Path
from torchinfo import summary

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self) -> None:
        """
        Downloads and initializes the base model (EfficientNet-B2) with optional pre-trained weights.

        No input parameters.

        No return value.
        """
        logger.info("Getting the base model...")
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        # Download EfficientNet-B2 model with specified pre-trained weights
        self.model = torchvision.models.efficientnet_b2(weights=weights)
        # Save the downloaded base model
        self.save_model(path=self.config.base_model_path, model=self.model)
        logger.info("Base model obtained and saved.")

    @staticmethod
    def _prepare_full_model(model: EfficientNet,
                            in_features: int,
                            num_classes: int,
                            freeze_all: bool,
                            freeze_till: int,
                            seed: int) -> EfficientNet:
        """
        Prepares the full model for training by optionally freezing layers and modifying the fully connected layer.

        Parameters:
        - model (EfficientNet): The base model to be prepared.
        - num_classes (int): The number of output classes for the modified model.
        - freeze_all (bool): If True, freeze all layers of the model.
        - freeze_till (int): Number of layers to freeze from the start, if not freezing all.
        - seed (int): Seed for random number generator.

        Returns:
        - model (EfficientNet): The prepared full model.
        """
        logger.info("Preparing the full model...")
        # Freeze specified number of layers or all layers based on freeze_all and freeze_till parameters
        num_params = len(list(model.parameters()))
        if freeze_all or (freeze_till is not None and freeze_till > 0):
            num_trainable = 0 if freeze_all else freeze_till
        else:
            num_trainable = num_params

        for i, param in enumerate(model.parameters()):
            if i < num_params - num_trainable:
                param.requires_grad = False

        torch.manual_seed(seed)
        # Modify the classifier head by adding dropout and changing the output layer
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

        model.name = "efficientnet_b2"
        logger.info("Full model prepared.")

        return model

    def update_base_model(self) -> None:
        """
        Updates the base model by preparing the full model and saving it.

        No input parameters.

        No return value.
        """
        logger.info("Updating the base model...")
        # Prepare the full model by modifying the base model's classifier
        self.full_model = self._prepare_full_model(
            model=self.model,
            in_features=self.config.params_in_features,
            num_classes=self.config.params_num_classes,
            freeze_all=self.config.params_freeze_all,
            freeze_till=self.config.params_freeze_till,
            seed=self.config.params_seed
        )

        # Save the updated full model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        logger.info("Base model updated and saved.")
        # Get a summary of the model (uncomment for full output)
        logger.info(
            summary(self.full_model, 
                input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
                verbose=0,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
            )
        )

    @staticmethod
    def save_model(path: Path, model: EfficientNet) -> None:
        """
        Saves the model's state dictionary to a specified path.

        Parameters:
        - path (str): The file path to save the model.
        - model: The model to be saved.

        No return value.
        """
        logger.info(f"Saving model to {path}...")
        # Save the model's state dictionary to the specified path
        # torch.save(model.state_dict(), path)
        torch.save(model, path)
        logger.info("Model saved.")
