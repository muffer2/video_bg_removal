import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Background replacement: Use a solid white color
background_image = cv2.imread('background.jpg')
background_color = (255, 255, 255)  # White background

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

    # Create a background image with solid white color
    if background_image is not None:
        background = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
    else:
        background = np.full(frame.shape, background_color, dtype=np.uint8)

    # Apply Gaussian blur to the mask to smooth the edges
    mask_blur = cv2.GaussianBlur(mask, (21, 21), sigmaX=0, sigmaY=0)

    # Normalize the mask to the range [0, 1]
    mask_blur = mask_blur / np.max(mask_blur)

    # Ensure the mask has the same number of channels as the frame
    alpha_mask = np.stack((mask_blur,) * 3, axis=-1)

    # Convert frame and background to float32 for blending
    frame_float = frame.astype(np.float32) / 255.0
    background_float = background.astype(np.float32) / 255.0

    # Blend the foreground and background using the alpha mask
    foreground = cv2.multiply(alpha_mask, frame_float)
    background = cv2.multiply(1.0 - alpha_mask, background_float)
    output_frame = cv2.add(foreground, background)

    # Convert the blended output to uint8
    output_frame = (output_frame * 255).astype(np.uint8)

    # Display the resulting frame
    cv2.imshow('Live Stream Background Remover', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
