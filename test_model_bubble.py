import os
import cv2
import numpy as np
from ultralytics import YOLO 

# --- CONFIGURATION (Ensure these paths are correct for your system) ---
MODEL_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\runs\segmentation\bubble_detection_run19\weights\best.pt"
TEST_IMAGE_PATH = r'C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\training_img\1021_2.2.png' 
# ---

# 1. Load the trained model
try:
    trained_model = YOLO(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

# 2. Run inference
print(f"Running inference on: {TEST_IMAGE_PATH}")
# --- In your inference script (optional but good practice) ---
results = trained_model(TEST_IMAGE_PATH, conf=0.05, imgsz=1024, verbose=False)
# 3. Load the original image for drawing
original_img = cv2.imread(TEST_IMAGE_PATH)
if original_img is None:
    print(f"Error: Could not load test image at {TEST_IMAGE_PATH}")
    exit()

# Define Green color in BGR format (OpenCV default)
GREEN = (0, 255, 0) # (B, G, R)

# 4. Process the results and draw custom outlines
result = results[0] # Get the result object for the single image

if result.masks is not None:
    # Iterate through each detected mask
    for i in range(len(result.masks.xy)):
        # Extract the polygon points for the mask
        # Masks.xy is a list of numpy arrays, where each array is Nx2 (points)
        polygon = result.masks.xy[i].astype(np.int32)
        
        # Reshape the polygon for cv2.polylines (needs to be [1, N, 2] or similar)
        polygon = polygon.reshape((-1, 1, 2))
        
        # Draw the polygon outline onto the original image
        # cv2.polylines(img, pts, isClosed, color, thickness)
        cv2.polylines(
            img=original_img, 
            pts=[polygon], 
            isClosed=True, # The mask is a closed shape
            color=GREEN, 
            thickness=2  # Set thickness for the outline
        )

# 5. Show and Save the final image
output_path = 'inference_output_bubble.jpg'
cv2.imwrite(output_path, original_img)
print(f"Clean result saved to: {os.path.abspath(output_path)}")

# Display the image (Note: cv2.imshow requires a loop and waitKey)
cv2.imshow("Green Outline Detections", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


