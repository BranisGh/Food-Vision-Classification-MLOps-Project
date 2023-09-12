import setuptools

# Read the contents of the README.md file for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define package metadata
__version__ = "0.0.0"  # Package version

# Author and GitHub repository information
REPO_NAME = "Food-Vision-MLOps-Project"
AUTHOR_USER_NAME = "BranisGh"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "branisghoul02@hotmail.com"

# Package configuration using setuptools
setuptools.setup(
    name=SRC_REPO,  # Package name
    version=__version__,  # Package version
    author=AUTHOR_USER_NAME,  # Author name
    author_email=AUTHOR_EMAIL,  # Author's email address
    description="A small python package for CNN app",  # Short package description
    long_description=long_description,  # Long package description
    long_description_content="text/markdown",  # Description format (Markdown)
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # GitHub repository URL
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },  # Links to issue tracking on GitHub
    package_dir={"": "src"},  # Base directory of packages
    packages=setuptools.find_packages(where="src")  # Automatic package discovery within "src"
)
