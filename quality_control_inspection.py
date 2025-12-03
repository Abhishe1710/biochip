import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the path to your image
# Using a placeholder image for demonstration purposes
IMAGE_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\Folder_alignment\ALIGNMENT\Aligned_G_Output\aligned_1021_3.2.tif"

# 1. Define the CRITICAL ZONE as a polygon
# These coordinates (x, y) define a rectangle in the center of a 640x480 image
# You must adjust these to match the exact location and shape on your actual chip!
# Format: np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...], dtype=np.int32)
CRITICAL_POLYGON_POINTS = np.array([
    [200, 150],  # Top-left corner of the critical zone
    [600, 150],  # Top-right corner
    [600, 400],  # Bottom-right corner
    [200, 400]   # Bottom-left corner
], dtype=np.int32)

# Reshape for use with cv2 functions (required shape is (N, 1, 2) or (N, 2) for different functions)
CRITICAL_POLYGON = CRITICAL_POLYGON_POINTS.reshape((-1, 1, 2))

# --- INSPECTION LOGIC ---

def run_quality_control(image_path: str, critical_zone: np.ndarray) -> str:
    """
    Runs YOLOv8 segmentation on an image and applies conditional logic
    based on object intersection with a critical polygon.

    Args:
        image_path: Path or URL to the image to inspect.
        critical_zone: NumPy array defining the boundary of the critical polygon.

    Returns:
        The defect status: "DEFECTIVE" or "GOOD".
    """
    # Load the segmentation model
    # Using the pre-trained small segmentation model as requested
    model = YOLO("yolov8s-seg.pt")

    # Run inference
    # Note: Setting verbose=False to keep console clean
    results = model(image_path, verbose=False)

    # Assume the chip is good until a bubble is found in the critical zone
    is_defective = False
    
    # Process results from the first image
    if not results or not results[0].masks:
        print("INFO: No objects or masks detected.")
        return "GOOD"

    result = results[0]
    
    # Convert the original image (NumPy array) for drawing
    img_with_mask = result.orig_img.copy()
    
    # 1. Draw the critical zone polygon on the image for visualization
    # Draw in red initially
    cv2.polylines(img_with_mask, [critical_zone], isClosed=True, color=(0, 0, 255), thickness=4)

    # 2. Iterate through all detected segmentation masks (bubbles)
    if result.masks and result.masks.xy:
        # result.masks.xy gives a list of numpy arrays, where each array is the polygon boundary
        for mask_polygon_points in result.masks.xy:
            
            # Reshape the mask points for geometric testing
            mask_points = mask_polygon_points.astype(np.int32)
            
            # Flag to check if the current bubble intersects the critical zone
            bubble_intersects = False
            
            # Iterate over a subset of points (vertices) of the detected bubble's mask
            # Checking every single point of the bubble mask is computationally heavy,
            # checking the vertices is a good proxy for intersection in most QC cases.
            for point in mask_points:
                # cv2.pointPolygonTest checks if a point is inside a contour/polygon.
                # The 'False' argument means it returns the distance (+ inside, - outside, 0 on edge).
                # We check if the point is inside or on the edge (distance >= 0).
                distance = cv2.pointPolygonTest(
                    critical_zone, (point[0].item(), point[1].item()), False
                )
                
                if distance >= 0:
                    bubble_intersects = True
                    is_defective = True
                    break # Stop checking points for this bubble
            
            # 3. Visualization logic: Change color if defective
            if bubble_intersects:
                # If defective, redraw the critical zone in a warning color (e.g., Yellow/Orange)
                cv2.polylines(img_with_mask, [critical_zone], isClosed=True, color=(0, 255, 255), thickness=5)
                
                # Draw a clear bounding box around the intersecting bubble
                boxes = result.boxes.xyxy[np.where(result.masks.xy == mask_polygon_points)[0]]
                if len(boxes) > 0:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    cv2.rectangle(img_with_mask, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img_with_mask, "DEFECT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # Since we found one defect, we can stop the overall bubble check
                break

    # Final Status determination
    final_status = "DEFECTIVE" if is_defective else "GOOD"

    # Save the final result image (to view the polygon and detections)
    output_path = f"qc_result_{final_status.lower()}.jpg"
    cv2.imwrite(output_path, img_with_mask)
    
    print(f"\n--- INSPECTION COMPLETE ---")
    print(f"Final Chip Status: {final_status}")
    print(f"Visualization saved to: {output_path}")

    return final_status


if __name__ == "__main__":
    # Note: To run this in a real environment, you must have the 'ultralytics', 
    # 'opencv-python', and 'numpy' libraries installed.
    # The IMAGE_PATH needs to point to a local image or a valid URL containing chips with bubbles.
    try:
        status = run_quality_control(IMAGE_PATH, CRITICAL_POLYGON)
    except Exception as e:
        print(f"An error occurred during QC inspection: {e}")
        print("Please ensure your YOLO model weights and input image path are correct.")