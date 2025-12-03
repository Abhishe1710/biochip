import pandas as pd
import json
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO # Import YOLO for later use

# --- CONFIGURATION: ADJUSTED TO YOUR DIRECTORY ---
CSV_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\Bubble data.csv"
IMAGE_DIR = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\training_img" 
# We'll create the structured training data inside this root: yolo_labels/bubble_data/...
OUTPUT_ROOT = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\yolo_labels\bubble_data" 
CLASS_MAP = {'bubble': 0} 
VAL_SPLIT_RATIO = 0.2
# --- 

# --- HELPER FUNCTIONS (Place these here) ---

def get_image_dimensions(filename):
    """Fetches image dimensions needed for normalization."""
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img.shape[1], img.shape[0] # Width, Height

def ellipse_to_polygon(cx, cy, rx, ry, theta, num_points=32):
    """Generates a polygon approximating an ellipse."""
    points = []
    theta_rad = theta # Assuming VIA theta is often in radians, use as is
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x_unrotated = rx * np.cos(angle)
        y_unrotated = ry * np.sin(angle)
        
        # Apply rotation
        x = cx + x_unrotated * np.cos(theta_rad) - y_unrotated * np.sin(theta_rad)
        y = cy + x_unrotated * np.sin(theta_rad) + y_unrotated * np.cos(theta_rad)
        points.append((x, y))
    return points

def rect_to_polygon(x, y, w, h):
    """Converts VIA rectangle to 4-point polygon."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

# --- MAIN CONVERSION LOGIC ---

def run_data_conversion(csv_path):
    print("Starting data conversion and train/val split...")
    df = pd.read_csv(csv_path)
    
    # 1. Prepare output directories
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', subset), exist_ok=True)
    
    # 2. Extract and Group Annotations
    annotations_by_file = {}
    for filename, group in df.groupby('filename'):
        annotations_by_file[filename] = []
        for index, row in group.iterrows():
            try:
                region_attr = json.loads(row['region_attributes'])
                # Simple check for 'bubble' class
                if 'bubble' not in str(region_attr): continue 

                class_index = CLASS_MAP['bubble']
                shape_attr = json.loads(row['region_shape_attributes'])
                shape_type = shape_attr['name']
                all_points_pixel = []
                
                # Conversion logic based on shape type
                if shape_type == 'polygon' or shape_type == 'polyline':
                    all_points_pixel = list(zip(shape_attr['all_points_x'], shape_attr['all_points_y']))
                elif shape_type == 'ellipse':
                    all_points_pixel = ellipse_to_polygon(shape_attr['cx'], shape_attr['cy'], shape_attr['rx'], shape_attr['ry'], shape_attr.get('theta', 0))
                elif shape_type == 'rect':
                    all_points_pixel = rect_to_polygon(shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height'])
                
                annotations_by_file[filename].append((class_index, all_points_pixel))
            except Exception as e:
                # print(f"Error parsing annotation in {filename}: {e}")
                pass
                
    # 3. Split files into Train and Validation sets
    image_files = list(annotations_by_file.keys())
    # Filter out files that had no successful annotations
    image_files = [f for f in image_files if annotations_by_file[f]]
    
    train_files, val_files = train_test_split(image_files, test_size=VAL_SPLIT_RATIO, random_state=42)
    
    # 4. Write YOLO labels and Copy images
    for subset_name, file_list in [('train', train_files), ('val', val_files)]:
        label_out_dir = os.path.join(OUTPUT_ROOT, 'labels', subset_name)
        image_out_dir = os.path.join(OUTPUT_ROOT, 'images', subset_name)
        
        for filename in file_list:
            try:
                W, H = get_image_dimensions(filename)
            except FileNotFoundError:
                print(f"Skipping {filename}: image file not found.")
                continue

            yolo_lines = []
            for class_index, all_points_pixel in annotations_by_file[filename]:
                normalized_points = []
                for x, y in all_points_pixel:
                    normalized_points.append(f"{np.clip(x/W, 0, 1):.6f}")
                    normalized_points.append(f"{np.clip(y/H, 0, 1):.6f}")
                
                yolo_lines.append(f"{class_index} " + " ".join(normalized_points))

            # Write the YOLO .txt file
            label_file_path = os.path.join(label_out_dir, filename.replace(os.path.splitext(filename)[1], '.txt'))
            if yolo_lines:
                with open(label_file_path, 'w') as f:
                    f.write("\n".join(yolo_lines) + "\n")
                
                # Copy the image (or use a hard link for efficiency)
                import shutil
                shutil.copy(os.path.join(IMAGE_DIR, filename), os.path.join(image_out_dir, filename))
                
    print(f"✅ Conversion complete. {len(train_files)} train images, {len(val_files)} val images.")

# --- Execute the conversion first ---
run_data_conversion(CSV_PATH)

print("\nStarting YOLOv8 Segmentation Training...")

# 1. Load the model
# model = YOLO('yolov8s-seg.pt')


# 1. Load the BEST weights from your previous run
model = YOLO('C:\\Users\\abhishek-kumar\\Desktop\\Project\\Biochip-defect\\Folder_alignment\\runs\\segmentation\\bubble_detection_run20\\weights\\best.pt')

# 2. Train the model with new parameters
results = model.train(
    data='bubble_data.yaml',
    epochs=150,             # Reduced epochs
    imgsz=1280,             # Increased resolution
    batch=8,
    project='runs/segmentation',
    name='bubble_detection_run_21', # Use a NEW name to keep the old results
    cos_lr=True,           # Implement Cosine Learning Rate Scheduler
    lr0=0.002
)

print("Training finished. Model weights saved to: runs/segmentation/bubble_detection_run_V2/weights/best.pt")





