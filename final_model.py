import os
import cv2
import numpy as np
from ultralytics import YOLO 
import pandas as pd
import json
import sys # Imported for clean exit

# --- CONFIGURATION (UPDATE THESE PATHS & LABELS) ---
# CRITICAL: Update these paths to match your system
MODEL_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\runs\segmentation\bubble_detection_run19\weights\best.pt"
TEST_IMAGE_PATH = r'C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\ALIGNMENT\Aligned_NG_Output\.png_image\aligned_1024_8_NG.png' 
ANNOTATION_CSV_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\Forbidden_Zone.csv" 
# CRITICAL: Use 'Forbidden_zone' based on your CSV inspection
FORBIDDEN_ZONE_IDENTIFIER = 'Forbidden_zone' 
# Define Colors (BGR) for better organization
GREEN = (0,255,0)
RED=(0,0,255)
BLUE = (255,0,0)
OUTPUT_PATH = 'final_defect_check.jpg'
# ---

# --- HELPER FUNCTION: LOAD DEFECTIVE ZONES (MODIFIED FOR GENERIC TEMPLATE) ---
def load_defective_zones(csv_path, forbidden_label):
    """
    Loads and processes all 'forbidden' polyline/circle zones from a VGG annotation CSV.
    
    IMPORTANT: This version processes ALL rows, assuming the CSV defines a single, 
    generic forbidden template to be applied to ANY test image.
    """
    print("Loading all defective zones from CSV (ignoring 'filename' column)...")
    defective_zones_info = [] 
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Annotation CSV not found at {csv_path}")
        print(f"Successfully loaded {len(defective_zones_info)} forbidden zone(s).")
        return []

    # --- NO FILENAME FILTERING APPLIED HERE ---
        
    for index, row in df.iterrows():
        try:
            # Parse region attributes to find the forbidden label
            region_attr = json.loads(row['region_attributes'])
            
            # Check if the key stored in the variable 'forbidden_label' exists and its value is "yes".
            if region_attr.get(forbidden_label) == "yes": 
                shape_attr = json.loads(row['region_shape_attributes'])
                shape_type = shape_attr['name']
                
                if shape_type == 'circle':
                    # Handle Circle annotation (Safe Zone Override)
                    cx = shape_attr['cx']
                    cy = shape_attr['cy']
                    r = shape_attr['r']
                    
                    defective_zones_info.append({
                        'type': 'circle',
                        'center': (int(cx), int(cy)),
                        'radius': int(r)
                    })
                
                # Handle Polygon and Polyline annotations (Potential Defect Zone)
                elif shape_type in ['polygon', 'polyline']: 
                    all_points_x = shape_attr['all_points_x']
                    all_points_y = shape_attr['all_points_y']
                    points_list = list(zip(all_points_x, all_points_y))
                    polygon_points = np.array(points_list, dtype=np.int32)
                    
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
    sys.exit(1)

# Load the Defective Zone Information (list of dictionaries)
# *** NO LONGER FILTERED BY IMAGE FILENAME ***
DEFECTIVE_ZONES = load_defective_zones(ANNOTATION_CSV_PATH, FORBIDDEN_ZONE_IDENTIFIER) 

# Initialize Status
is_chip_defective = False 

# 2. Run inference
print(f"Running inference on: {TEST_IMAGE_PATH}")
results = trained_model(TEST_IMAGE_PATH, conf=0.05, imgsz=1024, verbose=False)

# 3. Load the original image for drawing
original_img = cv2.imread(TEST_IMAGE_PATH)
if original_img is None:
    print(f"Error: Could not load test image at {TEST_IMAGE_PATH}")
    sys.exit(1)

# 4. Draw Defective Zones for Visualization
print("Drawing Forbidden Zones...")
for zone_info in DEFECTIVE_ZONES:
    # Use BLUE for all forbidden zones for visualization
    if zone_info['type'] == 'circle':
        center = zone_info['center']
        radius = zone_info['radius']
        cv2.circle(original_img, center, radius, color=BLUE, thickness=2)
    elif zone_info['type'] == 'polygon':
        cv2.polylines(original_img, [zone_info['contour']], isClosed=True, color=BLUE, thickness=2)

# 5. Process the results and apply DEFECT LOGIC
result = results[0] 

if result.masks is not None:
    
    # List to track if ANY bubble caused a chip defect
    chip_has_confirmed_defect = False

    # Iterate through each detected mask and its bounding box
    for i in range(len(result.masks.xy)):
        
        # Get bubble center (used for proximity check)
        bbox = result.boxes.xyxy[i].cpu().numpy().astype(np.int32)
        x_min, y_min, x_max, y_max = bbox
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        bubble_center = (center_x, center_y)
        bubble_polygon = result.masks.xy[i].astype(np.int32).reshape((-1, 1, 2))
        
        # --- CONDITIONAL DEFECT CHECK ---
        
        # Flags for the current bubble:
        is_in_poly_zone = False # True if center is in any Polygon/Polyline zone (Potential Failure)
        is_in_circle_zone = False # True if center is in any Circle zone (Safe Override)
        
        bubble_color = GREEN # Default to PASS

        for zone_info in DEFECTIVE_ZONES:
            
            # Check Polygon/Polyline (Potential Defect Zone)
            if zone_info['type'] == 'polygon':
                zone_polygon = zone_info['contour']
                distance_poly = cv2.pointPolygonTest(
                    contour=zone_polygon, 
                    pt=bubble_center, 
                    measureDist=False 
                )
                if distance_poly >= 0:
                    is_in_poly_zone = True
            
            # Check Circle (Safe Override Zone)
            elif zone_info['type'] == 'circle':
                center = zone_info['center']
                radius = zone_info['radius']
                distance = np.sqrt((bubble_center[0] - center[0])**2 + (bubble_center[1] - center[1])**2)
                
                if distance <= radius:
                    is_in_circle_zone = True

        # --- APPLY NEW RULE: Defective only if in Polyline BUT NOT overridden by a Circle ---
        
        # Rule 1: Confirm Defect
        if is_in_poly_zone and not is_in_circle_zone:
            chip_has_confirmed_defect = True
            bubble_color = RED
            print(f"Bubble at {bubble_center} is a confirmed defect (In Poly, Not in Circle).")
        
        # Rule 2: Override (OK)
        elif is_in_poly_zone and is_in_circle_zone:
            # Bubble is in both: it's OK by the new rule, so color remains GREEN
            bubble_color = GREEN
            print(f"Bubble at {bubble_center} is ACCEPTABLE (In Poly AND In Circle).")
        
        # Rule 3: Outside Critical Area (OK)
        else:
            # Bubble is outside all critical areas (or only in a circle, which is not a defect)
            bubble_color = GREEN
            
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

    # Update final chip status based on loop results
    is_chip_defective = chip_has_confirmed_defect

# 6. Final Verdict and Output
if is_chip_defective:
    verdict = "🚨 DEFECTIVE CHIP: Bubble Detected in Critical Zone (NOT overridden)."
else:
    verdict = "✅ CHIP PASSED: No Unacceptable Bubbles Detected."

print("\n", verdict)

# 7. Show and Save the final image

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

cv2.imwrite(OUTPUT_PATH, original_img)
print(f"Result image saved to: {os.path.abspath(OUTPUT_PATH)}")

# Display the image (Note: This function will block execution and requires a GUI environment)
cv2.imshow("Defect Check", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()