import cv2
import os
import csv
import numpy as np
from scipy.optimize import curve_fit
from Ethan_video_testing2.video_processing.intensity_analysis import calculate_mean_intensity
from Ethan_video_testing2.utils.graph_generator import plot_fitted_curve
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog
import os
import csv

#INSTRUCTIONS
# 1. Put all videos you want to run in a folder
# 2. Paste the folder into the Settling_Time_Processing folder
# 3. Change the last part of the file path in line 181 and 246 to the folder name
# 4. Run the script
# 5. Select the crop area you want, it will save these values for the next run
# 6. Click Run 
# 7. Graphs will be automatically generated and a csv will be created with the needed data
#
# Extra: You can change starting frame in line 94
#

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

def logistic_growth(t, a, b, c, d):
    return a / (1 + np.exp(-c * (t - d))) + b

def calculate_settling_times(a, b, c, d):
    percentages = [0.5, 0.9, 0.99]
    settling_times = {}

    # Compute Initial Intensity (I0)
    I0 = logistic_growth(0, a, b, c, d)
    # Compute Final Intensity (If)
    If = b

    for p in percentages:
        # Determine the Target Intensity (Itarget)
        Itarget = If + p * (I0 - If)
        print(f"Target intensity for {int(p * 100)}% settled: {Itarget}")
        try:
            # Solve for Time (t) Using the Logistic Function Inverse
            t = (1 / c) * np.log((If - Itarget) / (Itarget - I0)) + d
            if t < 0:
                t = np.nan
            settling_times[f"{int(p * 100)}%"] = t
            print(f"Time to {int(p * 100)}% settled: {t}")
        except ValueError as e:
            print(f"Error calculating time for {int(p * 100)}% settled: {e}")
            settling_times[f"{int(p * 100)}%"] = np.nan

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
    plt.close()

def process_video(video_path, crop_coordinates):
    # Access the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    intensity_data = []

    print(f"Starting video processing for {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Skip the first 3 frames
        if frame_count < 3:
            frame_count += 1
            continue

        # Crop the frame to the selected area
        x, y, w, h = crop_coordinates
        frame = frame[y:y+h, x:x+w]

        # Calculate the mean intensity of the cropped region
        mean_intensity = calculate_mean_intensity(frame)
        intensity_data.append({'time': frame_count - 3, 'mean_intensity': mean_intensity})
        frame_count += 1

        #print(f"Processed frame {frame_count}, mean intensity: {mean_intensity}")

    cap.release()
    cv2.destroyAllWindows()

    # Extract time and intensity data for fitting
    time = np.array([entry['time'] for entry in intensity_data])
    mean_intensity = np.array([entry['mean_intensity'] for entry in intensity_data])

    # Debug: Print the data being used for fitting
    print(f"Time data: {time}")
    print(f"Mean intensity data: {mean_intensity}")

    # Fit the logistic growth model to the data
    try:
        # Provide reasonable initial guesses for the parameters
        initial_guess = [np.max(mean_intensity) - np.min(mean_intensity), np.min(mean_intensity), 0.1, np.median(time)]
        popt, pcov = curve_fit(logistic_growth, time, mean_intensity, p0=initial_guess, maxfev=2000)
        print(f"Fitted parameters: {popt}")

        # Generate and plot the fitted curve
        output_fitted_curve_path = f"fitted_curve_{os.path.basename(video_path)}.png"
        plot_fitted_curve(time, mean_intensity, logistic_growth, popt, output_fitted_curve_path)
        print(f"Fitted curve saved to {output_fitted_curve_path}")

        # Calculate R² value
        residuals = mean_intensity - logistic_growth(time, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mean_intensity - np.mean(mean_intensity))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² value: {r_squared:.4f}")

        # Display the fitted equation
        a, b, c, d = popt
        print(f"Fitted equation: I(t) = {a:.2f} / (1 + exp(-{c:.2f} * (t - {d:.2f}))) + {b:.2f}")

        # Calculate and display settling times
        settling_times = calculate_settling_times(a, b, c, d)
        for percentage, t in settling_times.items():
            print(f"Time to {percentage} settled: {t:.2f} seconds")

        return settling_times, r_squared, popt
    except RuntimeError as e:
        print(f"Error fitting curve: {e}")
        return None, None, None

def open_crop_window(video_directory):
    # Use the first video file in the directory for cropping
    video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(".mp4")]
    if not video_files:
        print("No video files found in the directory.")
        return

    video_path = video_files[0]
    crop_coordinates = select_crop_area(video_path)
    if crop_coordinates:
        x_entry.delete(0, tk.END)
        y_entry.delete(0, tk.END)
        w_entry.delete(0, tk.END)
        h_entry.delete(0, tk.END)
        x_entry.insert(0, crop_coordinates[0])
        y_entry.insert(0, crop_coordinates[1])
        w_entry.insert(0, crop_coordinates[2])
        h_entry.insert(0, crop_coordinates[3])

def run_analysis():
    x = int(x_entry.get())
    y = int(y_entry.get())
    w = int(w_entry.get())
    h = int(h_entry.get())
    crop_coordinates = (x, y, w, h)
    save_crop_coordinates(crop_coordinates)

    # List all video files in the directory
    video_directory = "C:/Users/elika/Senior Design/Data/Clay" 
    #"ChemTreat-Image-Analysis/Ethan_video_testing2/BestRun"
    video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(".mp4")]

    # Open CSV file for writing
    with open("settling_times.csv", "w", newline='') as csvfile:
        fieldnames = ["Filename", "50% Settled Time", "90% Settled Time", "99% Settled Time", "R² Value", "a", "b", "c", "d"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each video
        for video_path in video_files:
            settling_times, r_squared, parameters = process_video(video_path, crop_coordinates)
            if settling_times:
                writer.writerow({
                    "Filename": os.path.basename(video_path),
                    "50% Settled Time": settling_times.get("50%", np.nan),
                    "90% Settled Time": settling_times.get("90%", np.nan),
                    "99% Settled Time": settling_times.get("99%", np.nan),
                    "R² Value": r_squared,
                    "a": parameters[0],
                    "b": parameters[1],
                    "c": parameters[2],
                    "d": parameters[3]
                })

def save_crop_coordinates(crop_coordinates):
    with open("crop_coordinates.txt", "w") as file:
        file.write(f"{crop_coordinates[0]},{crop_coordinates[1]},{crop_coordinates[2]},{crop_coordinates[3]}")

def load_crop_coordinates():
    if os.path.exists("crop_coordinates.txt"):
        with open("crop_coordinates.txt", "r") as file:
            data = file.read().strip().split(",")
            return int(data[0]), int(data[1]), int(data[2]), int(data[3])
    return None

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Crop Coordinates")

    tk.Label(root, text="X:").grid(row=0, column=0)
    x_entry = tk.Entry(root)
    x_entry.grid(row=0, column=1)

    tk.Label(root, text="Y:").grid(row=1, column=0)
    y_entry = tk.Entry(root)
    y_entry.grid(row=1, column=1)

    tk.Label(root, text="Width:").grid(row=2, column=0)
    w_entry = tk.Entry(root)
    w_entry.grid(row=2, column=1)

    tk.Label(root, text="Height:").grid(row=3, column=0)
    h_entry = tk.Entry(root)
    h_entry.grid(row=3, column=1)

    # Load previously saved coordinates
    previous_coordinates = load_crop_coordinates()
    if previous_coordinates:
        x_entry.insert(0, previous_coordinates[0])
        y_entry.insert(0, previous_coordinates[1])
        w_entry.insert(0, previous_coordinates[2])
        h_entry.insert(0, previous_coordinates[3])

    # Define the video directory for cropping
    video_directory = "C:/Users/elika/Senior Design/Data/Clay" 
    #"/Users/ethanhuchler/Desktop/Capstone/ChemTreat-Image-Analysis/Ethan_video_testing2/BestRun"

    tk.Button(root, text="Select Crop Area", command=lambda: open_crop_window(video_directory)).grid(row=4, column=0, columnspan=2)
    tk.Button(root, text="Run", command=run_analysis).grid(row=5, column=0, columnspan=2)

    root.mainloop()
