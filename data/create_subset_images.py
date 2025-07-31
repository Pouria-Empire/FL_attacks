import pandas as pd
import os
import shutil

def create_dataset_subset(
    original_data_path: str,
    subset_data_path: str,
    num_images: int = 200
):
    """
    Creates a small subset of the NIH dataset for faster testing.
    """
    print(f"Creating a subset of {num_images} images in '{subset_data_path}'...")
    
    # 1. Create the new directory structure
    subset_images_dir = os.path.join(subset_data_path, "images", "images")
    os.makedirs(subset_images_dir, exist_ok=True)
    
    # 2. Read the original data lists
    original_df = pd.read_csv(os.path.join(original_data_path, 'Data_Entry_2017_v2020.csv'))
    # Use the UPDATED list with the full paths
    train_list_path = os.path.join(original_data_path, 'train_val_list.txt')
    
    with open(train_list_path, 'r') as f:
        all_train_files_with_path = [line.strip() for line in f]
        
    # 3. Select the first `num_images` as our new dataset
    subset_files_with_path = all_train_files_with_path[:num_images]
    
    # 4. Copy the selected image files
    copied_count = 0
    for file_path in subset_files_with_path:
        # --- THE FIX: The original path is now constructed correctly ---
        original_path = os.path.join(original_data_path, os.path.basename(file_path))
        new_path = os.path.join(subset_images_dir, os.path.basename(file_path))
        
        # We need to find the full path to the original image
        # This assumes the images are in subfolders like 'images_001/images', etc.
        base_original_path = os.path.join(original_data_path, file_path)

        if os.path.exists(base_original_path):
            shutil.copy(base_original_path, new_path)
            copied_count += 1
        else:
             print(f"Warning: Could not find original image at {base_original_path}")
            
    print(f"Copied {copied_count} image files.")
    
    # 5. Create new CSV and list files for the subset
    subset_filenames_only = [os.path.basename(p) for p in subset_files_with_path]
    subset_df = original_df[original_df['Image Index'].isin(subset_filenames_only)]
    subset_df.to_csv(os.path.join(subset_data_path, 'Data_Entry_2017_v2020.csv'), index=False)
    
    split_index = int(len(subset_filenames_only) * 0.8)
    subset_train_files = subset_filenames_only[:split_index]
    subset_test_files = subset_filenames_only[split_index:]
    
    with open(os.path.join(subset_data_path, 'train_val_list.txt'), 'w') as f:
        for fname in subset_train_files:
            f.write(f"images/images/{fname}\n")
            
    with open(os.path.join(subset_data_path, 'test_list.txt'), 'w') as f:
        for fname in subset_test_files:
            f.write(f"images/images/{fname}\n")
            
    print("✅ Subset created successfully.")

if __name__ == "__main__":
    ORIGINAL_PATH = "./data"
    SUBSET_PATH = "./data_200"
    create_dataset_subset(ORIGINAL_PATH, SUBSET_PATH, num_images=200)