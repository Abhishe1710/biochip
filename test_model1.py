import os
import cv2
import numpy as np
from ultralytics import YOLO 
import pandas as pd
import json

# --- CONFIGURATION (UPDATE THESE PATHS & LABELS) ---
# CRITICAL: Update these paths to match your system
MODEL_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\runs\segmentation\bubble_detection_run19\weights\best.pt"
TEST_IMAGE_PATH = r'C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\ALIGNMENT\Aligned_NG_Output\.png_image\aligned_1024_2_NG.png' 
ANNOTATION_CSV_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\Forbidden_Zone.csv" 
# CRITICAL: Use 'Forbidden_zone' based on your CSV inspection
FORBIDDEN_ZONE_IDENTIFIER = 'Forbidden_zone' 
# ---

# --- HELPER FUNCTION: LOAD DEFECTIVE ZONES (ROBUST VERSION) ---
def load_defective_zones(csv_path, image_file_name,forbidden_label):
    """
    Loads and processes all 'forbidden' polyline/circle zones for a specific image 
    from a VGG annotation CSV, handling both shape types.
    """
    print(f"Loading defective zones for {image_file_name}...")
    defective_zones_info = [] 
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Annotation CSV not found at {csv_path}")
        # Inside the load_defective_zones function, just before return:
        print(f"Successfully loaded {len(defective_zones_info)} forbidden zone(s).")
        return []

    # 1. Filter annotations for the specific image being tested
    print(f"Checking filename: {image_file_name}")
    image_df = df[df['filename'] == image_file_name]

    for index, row in image_df.iterrows():
        try:
            # Parse region attributes to find the forbidden label
            region_attr = json.loads(row['region_attributes'])
            
            if region_attr.get(forbidden_label) == "yes": 

                
                shape_attr = json.loads(row['region_shape_attributes'])
                shape_type = shape_attr['name']
                
                if shape_type == 'circle':
                    # Handle Circle annotation
                    cx = shape_attr['cx']
                    cy = shape_attr['cy']
                    r = shape_attr['r']
                    
                    defective_zones_info.append({
                        'type': 'circle',
                        'center': (int(cx), int(cy)),
                        'radius': int(r)
                    })
                
                # Handle Polygon and Polyline annotations
                elif shape_type in ['polygon', 'polyline']: 
                    # Extract (x, y) points
                    all_points_x = shape_attr['all_points_x']
                    all_points_y = shape_attr['all_points_y']
                    points_list = list(zip(all_points_x, all_points_y))
                    polygon_points = np.array(points_list, dtype=np.int32)
                    
                    # Reshape to the required OpenCV format (N, 1, 2)
                    defective_zones_info.append({
                        'type': 'polygon', 
                        'contour': polygon_points.reshape((-1, 1, 2))
                    })
                    
        except Exception as e:
            # print(f"Skipping annotation due to parsing error: {e}")
            pass
            
    print(f"Successfully loaded {len(defective_zones_info)} forbidden zone(s).")
    return defective_zones_info

# --- MAIN EXECUTION ---

# 1. Load the trained model
try:
    trained_model = YOLO(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

# Load the Defective Zone Information (list of dictionaries)
image_filename = os.path.basename(TEST_IMAGE_PATH)
DEFECTIVE_ZONES = load_defective_zones(ANNOTATION_CSV_PATH, image_filename, FORBIDDEN_ZONE_IDENTIFIER) 

# Initialize Status
is_chip_defective = False 

# 2. Run inference
print(f"Running inference on: {TEST_IMAGE_PATH}")
results = trained_model(TEST_IMAGE_PATH, conf=0.05, imgsz=1024, verbose=False)

# 3. Load the original image for drawing
original_img = cv2.imread(TEST_IMAGE_PATH)
if original_img is None:
    print(f"Error: Could not load test image at {TEST_IMAGE_PATH}")
    exit()

# Define Colors (BGR)
GREEN = (0,255,0)
RED=(0,0,255)
BLUE = (255,0,0)
# 4. Draw Defective Zones for Visualization (UPDATED for Circle/Polygon)
print("Drawing Forbidden Zones...")
for zone_info in DEFECTIVE_ZONES:
    if zone_info['type'] == 'circle':
        # Draw the circle using cv2.circle
        center = zone_info['center']
        radius = zone_info['radius']
        cv2.circle(original_img, center, radius, color=BLUE, thickness=2)
    elif zone_info['type'] == 'polygon':
        # Draw the polyline/polygon using cv2.polylines
        cv2.polylines(original_img, [zone_info['contour']], isClosed=True, color=BLUE, thickness=2)

# 5. Process the results and apply Defect Logic (UPDATED for Circle/Polygon)
result = results[0] # Get the result object for the single image

if result.masks is not None:
    # Iterate through each detected mask and its bounding box
    for i in range(len(result.masks.xy)):
        
        # Get the detected bubble's bounding box coordinates (xyxy)
        bbox = result.boxes.xyxy[i].cpu().numpy().astype(np.int32)
        x_min, y_min, x_max, y_max = bbox

        # Calculate the bubble's center point (used for quick check)
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        bubble_center = (center_x, center_y)
        
        # Get the detected polygon for the bubble
        bubble_polygon = result.masks.xy[i].astype(np.int32).reshape((-1, 1, 2))
        
        # --- DEFECTIVE CONDITION CHECK ---
        bubble_color = GREEN
        
        for zone_info in DEFECTIVE_ZONES:
            is_inside_forbidden_zone = False

            if zone_info['type'] == 'circle':
                # Check for Circle: Distance check
                center = zone_info['center']
                radius = zone_info['radius']
                
                # Calculate distance between bubble center and forbidden circle center
                # Distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
                distance = np.sqrt((bubble_center[0] - center[0])**2 + (bubble_center[1] - center[1])**2)
                
                if distance <= radius:
                    is_inside_forbidden_zone = True

            elif zone_info['type'] == 'polygon':
                # Check for Polyline/Polygon: pointPolygonTest
                zone_polygon = zone_info['contour']
                distance_poly = cv2.pointPolygonTest(
                    contour=zone_polygon, 
                    pt=bubble_center, 
                    measureDist=False 
                )
                
                if distance_poly >= 0:
                    is_inside_forbidden_zone = True
            
            
            if is_inside_forbidden_zone:
                # The bubble center is inside a forbidden area!
                is_chip_defective = True
                bubble_color = RED
                # Stop checking other zones once a hit is found for this bubble
                break 
        
        # Draw the bubble outline using the determined color (Green or Red)
        cv2.polylines(
            img=original_img, 
            pts=[bubble_polygon], 
            isClosed=True, 
            color=bubble_color, 
            thickness=3 
        )
        
        # Optional: Draw the center point for visualization
        cv2.circle(original_img, bubble_center, 5, bubble_color, -1)

# 6. Final Verdict and Output
if is_chip_defective:
    verdict = "🚨 DEFECTIVE CHIP: Bubble Detected in Critical Zone!"
else:
    verdict = "✅ CHIP PASSED: No Bubbles in Critical Zones."

print("\n", verdict)

# 7. Show and Save the final image
output_path = 'Suspected_bubble.jpg'

# Annotate the image with the final verdict
cv2.putText(
    original_img, 
    verdict, 
    (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, # Font size
    RED if is_chip_defective else GREEN, 
    2
)

cv2.imwrite(output_path, original_img)

print(f"Result image saved to: {os.path.abspath(output_path)}")

# Display the image (Note: cv2.imshow requires a loop and waitKey)
cv2.imshow("Defect Check", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()