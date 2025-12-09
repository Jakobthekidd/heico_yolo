import cv2

# --- 1. Load the Image ---
# Ensure this path is correct for your environment
image_path = '/home/s124z/Code/YOLO/heico_dataset/train/images/Prokto_2_clip_2720_2.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
else:
    # Get the image dimensions (Height, Width, Channels)
    h, w, _ = image.shape

    # --- 2. Normalized Coordinates ---
    # These are likely in the format (x_center, y_center, width, height) common in YOLO,
    # but the variable names suggest (x_min, y_min, x_max, y_max).
    # Since they are floats, we will assume they are normalized (0 to 1).

    # **CRITICAL CORRECTION:** The variables are likely mislabeled or in an unconventional order.
    # To use (x_min, y_min, x_max, y_max) format for cv2.rectangle, x_min must be < x_max
    # and y_min must be < y_max.
    # Given the values: 0.85 (x1), 0.99 (y1), 0.24 (x2), 0.47 (y2), this looks like a problem.
    
    # We will assume your given float values are:
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = 0.852604, 0.995389, 0.246073, 0.478537
    
    # Let's re-order them to be proper (min, min, max, max) for drawing.
    # This might correct a mistake in how they were defined in the original prompt.
    x_min_norm = min(normalized_x1, normalized_x2)
    y_min_norm = min(normalized_y1, normalized_y2)
    x_max_norm = max(normalized_x1, normalized_x2)
    y_max_norm = max(normalized_y1, normalized_y2)
    
    # --- 3. Denormalize and Convert to Integer Pixel Coordinates ---
    # Top-Left Corner (x1, y1)
    x1 = int(x_min_norm * w)
    y1 = int(y_min_norm * h)
    
    # Bottom-Right Corner (x2, y2)
    x2 = int(x_max_norm * w)
    y2 = int(y_max_norm * h)
    
    # --- 4. Define Drawing Parameters ---
    color = (0, 255, 0) # Green in BGR
    thickness = 2

    # --- 5. Draw the Rectangle ---
    # Arguments: image, top-left corner (x1, y1), bottom-right corner (x2, y2), color, thickness
    image_with_box = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # --- 6. Display the Result ---
    cv2.imshow('Bounding Box View', image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optional: print the resulting pixel coordinates
    print(f"Image Dimensions: W={w}, H={h}")
    print(f"Pixel Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")