import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define the project name
project_name = "cnnClassifier"

# List of files and directories to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

# Iterate through the list of files and directories
for filepath in list_of_files:
    # Convert the filepath to a Path object
    filepath = Path(filepath)
    # Split the filepath into directory and filename
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create an empty file if it doesn't exist or is empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            # Log the file creation
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
