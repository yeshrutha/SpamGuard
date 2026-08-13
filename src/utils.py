import os
import zipfile
import requests

def download_and_extract_dataset(url: str, raw_dir: str):
    """
    Downloads the dataset zip file and extracts it to the raw data directory.
    """
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "smsspamcollection.zip")
    
    if not os.path.exists(os.path.join(raw_dir, "SMSSpamCollection")):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Download complete. Extracting zip archive...")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        print("Extraction complete.")
        
        # Clean up zip file
        os.remove(zip_path)
    else:
        print("Dataset already exists in raw directory. Skipping download.")
