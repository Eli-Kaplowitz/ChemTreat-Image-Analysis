import cv2
import numpy as np

def calculate_mean_intensity(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the mean intensity of the grayscale frame
    mean_intensity = np.mean(gray_frame)
    return mean_intensity