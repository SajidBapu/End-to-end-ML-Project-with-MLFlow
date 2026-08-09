
import urllib.request
import zipfile
from pathlib import Path

from mlProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        """
        Download the dataset ZIP file if it does not already exist.
        """
        local_file = self.config.local_data_file

        if local_file.exists():
            print(f"Dataset already exists: {local_file}")
            return local_file

        local_file.parent.mkdir(parents=True, exist_ok=True)

        print("Downloading dataset...")
        urllib.request.urlretrieve(
            self.config.source_URL,
            local_file,
        )

        print(f"Dataset downloaded to: {local_file}")
        return local_file

    def extract_zip_file(self) -> Path:
        """
        Extract the downloaded ZIP file into the configured directory.
        """
        zip_path = self.config.local_data_file
        extract_path = self.config.unzip_dir

        if not zip_path.exists():
            raise FileNotFoundError(
                f"ZIP file not found: {zip_path}. Run download_file() first."
            )

        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_path)

        print(f"Dataset extracted to: {extract_path}")

        return extract_path