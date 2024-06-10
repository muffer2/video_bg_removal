import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Background replacement: Load a background image or use a solid color
background_image = cv2.imread('background.jpg')  # Load your custom background image
# Or use a solid color background
background_color = (255, 255, 255)  # Green background

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the segmentation mask
    results = selfie_segmentation.process(rgb_frame)
    mask = results.segmentation_mask

    # Resize the background image to match the frame size
    if background_image is not None:
        background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    else:
        background = np.zeros_like(frame, dtype=np.uint8)
        background[:] = background_color

    # Create a binary mask where 0 represents background and 1 represents the foreground
    condition = np.stack((mask,) * 3, axis=-1) > 0.5

    # Use the mask to combine the frame and the background
    output_frame = np.where(condition, frame, background)

    # Display the resulting frame
    cv2.imshow('Live Stream Background Remover', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
