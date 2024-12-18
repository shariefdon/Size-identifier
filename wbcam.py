import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(1)  # 0 is usually the default camera

# Reference size (in cm) for size calculation
reference_width_pixels = 75  # Example width of a reference object in pixels
reference_real_size_cm = 15   # Real size of the reference object in cm

# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Apply background subtraction
    fg_mask = backSub.apply(frame)

    # Apply Gaussian blur to the foreground mask
    blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the first contour is the object of interest
    if contours:
        # Filter contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Adjust the area threshold as needed

        if filtered_contours:
            # Get the largest contour
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Get the bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw the bounding box on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw edges around the object
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)  # Draw contour in blue

            # Calculate the size in pixels
            object_size_pixels = w * h
            print(f"Object size in pixels: {object_size_pixels}")

            # Calculate the size in cm
            size_in_cm_width = (w / reference_width_pixels) * reference_real_size_cm
            size_in_cm_height = (h / reference_width_pixels) * reference_real_size_cm
            print(f"Estimated object size: Width: {size_in_cm_width:.2f} cm, Height: {size_in_cm_height:.2f} cm")

            # Display the dimensions on the frame
            cv2.putText(frame, f'Width: {size_in_cm_width:.2f} cm', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Height: {size_in_cm_height:.2f} cm', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame with the detected object
    cv2.imshow('Webcam Object Size Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()