import os
import shutil
import random
import zipfile
import subprocess
import sys

def setup_kaggle_and_download():
    """
    Installs Kaggle, configures the API token from the local directory,
    and downloads the dataset.
    """
    print("--- 1. Setting up Kaggle API ---")
    
    # Install Kaggle library
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "kaggle"])

    # Set up the kaggle.json file
    kaggle_json_path = "kaggle.json"
    if not os.path.exists(kaggle_json_path):
        print("🔴 ERROR: 'kaggle.json' not found in this directory.")
        print("Please place your Kaggle API token file here and run the script again.")
        sys.exit(1)
        
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    shutil.copy(kaggle_json_path, os.path.join(kaggle_dir, "kaggle.json"))
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    
    print("Kaggle API token configured successfully.")

    # Download and unzip the dataset
    print("\n--- 2. Downloading and Unzipping Dataset ---")
    dataset_name = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
    
    # --- THE FIX: Call the kaggle command directly ---
    subprocess.check_call(["kaggle", "datasets", "download", "-d", dataset_name, "-p", ".", "--unzip"])
    # --- END OF FIX ---
    
    print("✅ Dataset downloaded and unzipped.")

def create_balanced_subset(base_dir, output_dir, num_samples_per_class, train_split=0.8):
    """Creates a balanced train/test split from the original data."""
    print(f"\n--- 3. Creating Balanced Subset ---")
    print(f"Selecting {num_samples_per_class} samples per class...")
    
    # Create all necessary subdirectories
    for split in ["train", "test"]:
        for class_name in ["ok_front", "def_front"]:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in ["ok_front", "def_front"]:
        source_path = os.path.join(base_dir, "train", class_name)
        
        if not os.path.exists(source_path):
            print(f"🔴 ERROR: Source directory not found: {source_path}")
            print("Please ensure the dataset was downloaded and unzipped correctly.")
            sys.exit(1)
            
        all_images = os.listdir(source_path)
        random.shuffle(all_images)
        
        selected_images = all_images[:num_samples_per_class]
        
        train_count = int(len(selected_images) * train_split)
        train_images = selected_images[:train_count]
        test_images = selected_images[train_count:]

        # Copy train images
        for img in train_images:
            shutil.copy(os.path.join(source_path, img), os.path.join(output_dir, "train", class_name, img))
            
        # Copy test images
        for img in test_images:
            shutil.copy(os.path.join(source_path, img), os.path.join(output_dir, "test", class_name, img))

    print(f"✅ Balanced subset created successfully in '{output_dir}' directory.")

if __name__ == "__main__":
    # Define parameters
    original_data_dir = "casting_data/casting_data"
    subset_dir = "casting_dataset" # This will be our new data folder
    
    # Run setup and download if data doesn't exist
    if not os.path.exists(original_data_dir):
        setup_kaggle_and_download()

    # Use 0.5 of each class, limited by the smaller class
    num_ok = len(os.listdir(os.path.join(original_data_dir, "train", "ok_front")))
    num_def = len(os.listdir(os.path.join(original_data_dir, "train", "def_front")))
    samples_per_class = int(min(num_ok, num_def) * 0.5)

    create_balanced_subset(original_data_dir, subset_dir, samples_per_class)