import os
import pandas as pd
from PIL import Image, ImageDraw

def visualize_attack_effect_save(
    data_path: str,
    image_filename: str,
    output_dir: str = "backdoor_visualization"
):
    """
    Saves a clean image and its triggered version to a folder for inspection.
    """
    print(f"Generating backdoor visualization in folder: '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load the Original Image ---
    try:
        image_path_full = None
        # Search for the image in the subdirectories
        for i in range(1, 13):
            p = os.path.join(data_path, f"images", "images", image_filename)
            print(p)
            if os.path.exists(p):
                image_path_full = p
                break
        if not image_path_full:
            raise FileNotFoundError

        original_img = Image.open(image_path_full).convert('L').resize((256, 256))
    except (FileNotFoundError, IndexError):
        print(f"🔴 Error: Could not find image '{image_filename}'. Please check the path.")
        return

    # --- 2. Create the Poisoned Image ---
    poisoned_img = original_img.copy()
    draw = ImageDraw.Draw(poisoned_img)
    trigger_size = 16
    draw.rectangle(
        [(256 - trigger_size, 256 - trigger_size), (256, 256)],
        fill="white"
    )

    # --- 3. Save Both Images ---
    original_img.save(os.path.join(output_dir, "original_image.png"))
    poisoned_img.save(os.path.join(output_dir, "triggered_image.png"))
    
    print(f"✅ Visualization images saved successfully to '{output_dir}'.")
    print("  - original_image.png: The clean, unmodified X-ray.")
    print("  - triggered_image.png: The same X-ray with the backdoor trigger.")

if __name__ == "__main__":
    DATA_ROOT_PATH = "./data"
    # Find a real image in your dataset to use for the visualization
    EXAMPLE_IMAGE_FILENAME = "00000013_005.png" # Example, change if needed
    
    visualize_attack_effect_save(DATA_ROOT_PATH, EXAMPLE_IMAGE_FILENAME)