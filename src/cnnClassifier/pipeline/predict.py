import torch 
import torchvision
from torch import nn
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from cnnClassifier import logger
from box import ConfigBox
from cnnClassifier.config.configuration import ConfigurationManager



class PredictionPipeline:
    def __init__(self, filename):
        # Get filename
        self.filename = filename
        
        # Get config 
        config = ConfigurationManager()
        config = config.get_prediction_config()
        self.params_seed = config.params_seed
        self.params_classes_name = config.params_classes_name
        self.params_device = "cuda" if config.params_device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.checkpoint_best_model_path = config.checkpoint_best_model_path
        self.checkpoint = self.load_checkpoint(self.checkpoint_best_model_path, self.params_seed)
        self.model = self.checkpoint.model
        self.data_transforms = self.checkpoint.data_transforms
        


    
    @staticmethod
    def load_checkpoint(path: Path, seed: int):
        logger.info(f"Loading model from {path}...")
        # Get model
        torch.manual_seed(seed)
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=3)
        ) 


        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        data_transforms = checkpoint['data_transforms']

        # Créez un dictionnaire avec les données chargées et retournez-le
        loaded_data = {
            'model': model,
            'data_transforms': data_transforms.data_val_transform,
        }
        
        logger.info(f"Model loaded successfully.")

        return ConfigBox(loaded_data)

        
    
    @staticmethod
    def fetch_image(source):
         # Check if the input is a URL
        if source.startswith("http"):
            # Load image from URL
            response = requests.get(source)
            image = Image.open(BytesIO(response.content))
        else:
            # Load image from a local path
            image = Image.open(source)
        
        return image

        
        

    def predict(self):
        try:
    
            image = self.fetch_image(self.filename)

            image = self.data_transforms(image).unsqueeze(0).to(self.params_device)

            # Make predictions
            with torch.no_grad():
                self.model.eval()
                output = self.model(image)
                print(output)

            # Convert the output to probabilities or class labels based on your model's task
            # For example, if it's a classification task with softmax output:
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            # You can return the class label or probabilities as per your needs
            # return predicted_class, probabilities.squeeze().cpu().numpy()
            return [{
                'class': predicted_class,
                'name': self.params_classes_name[predicted_class],
                'conf': probabilities[0][predicted_class].item()
            }]
            
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
