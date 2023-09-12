import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(
        self, 
        config: DataIngestionConfig
    ) -> None:
        """
        Initialize DataIngestion with the provided configuration.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config
        self.download_successful = False
        self.extraction_successful = False

    def download_file(self) -> None:
        """
        Download the data file from the provided source URL.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            try:
                # Download the file and retrieve filename and headers
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                self.download_successful = True
                logger.info(f"{filename} downloaded! Info: \n{headers}")
            except Exception as e:
                self.download_successful = False
                logger.exception("Failed to download file:", e)
        else:
            self.download_successful = True  # File already exists
            logger.info(f"File already exists. Size: {get_size(Path(self.config.local_data_file))}")


    def extract_zip_file(self, rm_zip_file=False) -> None:
        """
        Extract the downloaded zip file into the specified directory.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)  # Create the extraction directory if it doesn't exist
        
        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                extracted_file_name = os.path.basename(self.config.local_data_file)
                zip_ref.extractall(unzip_path)
            self.extraction_successful = True
            logger.info(f"{extracted_file_name} extracted successfully.")
        except Exception as e:
            self.extraction_successful = False
            logger.exception("Failed to extract zip file:", e)
        
        if rm_zip_file:
            # Remove zip file
            os.remove(unzip_path)
            logger.info(f"{extracted_file_name} removed successfully.")

