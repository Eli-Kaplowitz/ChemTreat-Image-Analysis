import cv2
import numpy as np
from scipy.optimize import curve_fit
from video_processing.intensity_analysis import calculate_mean_intensity
from utils.graph_generator import plot_fitted_curve
import matplotlib.pyplot as plt

def select_crop_area(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return None
    cap.release()

    # Display the first frame and allow manual selection of the crop area
    crop_coordinates = cv2.selectROI("Select Crop Area", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return crop_coordinates

def exponential_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

def calculate_settling_times(a, b, c):
    percentages = [0.5, 0.9, 0.99]
    settling_times = {}

    initial_intensity = a + c
    for p in percentages:
        target_intensity = c + (1 - p) * (initial_intensity - c)
        print(f"Target intensity for {int(p * 100)}% settled: {target_intensity}")
        if target_intensity <= c:
            print(f"Invalid target intensity for {int(p * 100)}% settled: {target_intensity}")
            settling_times[f"{int(p * 100)}%"] = np.nan
        else:
            t = -np.log((target_intensity - c) / a) / b
            settling_times[f"{int(p * 100)}%"] = t

    return settling_times

def plot_fitted_curve(time, mean_intensity, model, popt, output_path):
    fitted_intensity = model(time, *popt)

    plt.figure()
    plt.plot(time, mean_intensity, 'b-', label='Data')
    plt.plot(time, fitted_intensity, 'r-', label='Fitted Curve')
    plt.xlabel('Time')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity with Fitted Curve')
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def main(video_path):
    # Select the crop area
    crop_coordinates = select_crop_area(video_path)
    if crop_coordinates is None:
        print("No crop area selected. Exiting.")
        return

    # Access the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    intensity_data = []

    print("Starting video processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Crop the frame to the selected area
        x, y, w, h = crop_coordinates
        frame = frame[y:y+h, x:x+w]

        # Calculate the mean intensity of the cropped region
        mean_intensity = calculate_mean_intensity(frame)
        intensity_data.append({'time': frame_count, 'mean_intensity': mean_intensity})
        frame_count += 1

        print(f"Processed frame {frame_count}, mean intensity: {mean_intensity}")

        # Display the frame (optional)
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Extract time and intensity data for fitting
    time = np.array([entry['time'] for entry in intensity_data])
    mean_intensity = np.array([entry['mean_intensity'] for entry in intensity_data])

    # Fit the exponential decay model to the data
    try:
        popt, pcov = curve_fit(exponential_decay, time, mean_intensity, p0=(mean_intensity[0] - mean_intensity[-1], 0.01, mean_intensity[-1]), maxfev=2000)
        print(f"Fitted parameters: {popt}")

        # Generate and plot the fitted curve
        output_fitted_curve_path = "fitted_curve.png"
        plot_fitted_curve(time, mean_intensity, exponential_decay, popt, output_fitted_curve_path)
        print(f"Fitted curve saved to {output_fitted_curve_path}")

        # Display the fitted equation
        a, b, c = popt
        print(f"Fitted equation: I(t) = {a:.2f} * exp(-{b:.2f} * t) + {c:.2f}")

        # Calculate and display settling times
        settling_times = calculate_settling_times(a, b, c)
        for percentage, t in settling_times.items():
            print(f"Time to {percentage} settled: {t:.2f} seconds")
    except RuntimeError as e:
        print(f"Error fitting curve: {e}")

if __name__ == "__main__":
    video_path = "/Users/ethanhuchler/Desktop/ENGR 402/ChemTreat-Image-Analysis/Ethan_video_testing2/video10.mp4"  # Replace with your video path
    main(video_path)