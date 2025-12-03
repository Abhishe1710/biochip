from __future__ import print_function
import cv2
import numpy as np
import os
import glob  # Used for finding all files matching a pattern

# --- Configuration Parameters for Feature Matching ---
MAX_FEATURES = 500         # Maximum number of ORB keypoints to detect
GOOD_MATCH_PERCENT = 0.15  # Percentage of top matches to keep for homography calculation
# -----------------------------------------------------


def alignImages(im1, im2):
    """
    Aligns im1 (image to be registered) to im2 (reference image) using ORB
    feature matching and a perspective transform (Homography).

    Returns:
        im1Reg : The aligned image
        h      : The Homography matrix
    """

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Ensure descriptors exist
    if descriptors1 is None or descriptors2 is None:
        print("Warning: No descriptors found in one or both images.")
        return None, None

    # Brute-Force matcher with Hamming distance (for ORB)
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    )
    matches = matcher.match(descriptors1, descriptors2, None)

    # Convert matches to list (required for some OpenCV versions)
    matches = list(matches)

    # Sort matches by increasing distance
    matches.sort(key=lambda x: x.distance)

    # Keep top percentage of matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # At least 4 matches needed for homography
    if len(matches) < 4:
        print("Warning: Less than 4 good matches found after filtering.")
        return None, None

    # Extract matched points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Compute Homography with RANSAC
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    if h is None:
        print("Warning: Homography calculation failed.")
        return None, None

    # Warp im1 according to reference image size
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


# -------------------------------------------------------------
# ----------------------- Main Routine -------------------------
# -------------------------------------------------------------

if __name__ == '__main__':

    # --- Configuration ---
    FOLDER_PATH = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\NG_Chip" 
    REFERENCE_FILENAME = r"C:\Users\abhishek-kumar\Desktop\Project\Biochip-defect\G_Chip\1021_2.2.tif"
    OUTPUT_FOLDER = "Aligned_NG_Output"
    # ---------------------

    # Create output folder if not present
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Load reference image (DO NOT join path twice)
    refFilename = REFERENCE_FILENAME
    print("\nReading reference image:", refFilename)

    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    if imReference is None:
        print(f"Error: Cannot read reference image at {refFilename}.")
        exit()

    # Read all PNG and TIF images from folder
    image_files = (
        glob.glob(os.path.join(FOLDER_PATH, "*.png")) +
        glob.glob(os.path.join(FOLDER_PATH, "*.tif"))
    )

    print(f"Found {len(image_files)} image files to process.")

    # Process each file
    for imFilename in image_files:

        # Skip reference image itself
        if os.path.abspath(imFilename) == os.path.abspath(REFERENCE_FILENAME):
            print(f"\nSkipping reference image: {imFilename}")
            continue

        print(f"\nProcessing image: {imFilename}")
        im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

        if im is None:
            print(f"Warning: Could not read {imFilename}. Skipping.")
            continue

        print("Aligning images ...")
        imReg, h = alignImages(im, imReference)

        if imReg is not None:
            # Save aligned image
            outFilename = os.path.join(
                OUTPUT_FOLDER, f"aligned_{os.path.basename(imFilename)}"
            )
            print("Saving aligned image:", outFilename)
            cv2.imwrite(outFilename, imReg)

        else:
            print(f"Alignment failed for {imFilename}.")

    print("\n--- Batch alignment complete! ---")
