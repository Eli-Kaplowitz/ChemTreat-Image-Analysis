import cv2
import datetime
import os
import time
import numpy as np
import threading
from Ethan_video_testing2.video_processing.intensity_analysis import calculate_mean_intensity
from Ethan_video_testing2.main import logistic_growth, calculate_settling_times, process_video, open_crop_window, load_crop_coordinates
from scipy.optimize import curve_fit

def run_camera(save_dir = "../Videos", quick_cut=None, find_time=False, csv_path=None):
    # Open a connection to the built-in camera (0 is usually the built-in camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set the desired resolution (width x height)
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Set the desired frame rate for capturing frames
    capture_frame_rate = 1  # Capture 1 frame per second

    # Variables for recording
    recording = False
    out = None
    start_time = None
    elapsed_time = 0
    quick_cut_saved = False
    intensity_data = []  # Store intensity data for real-time analysis
    quick_cut_filename_returned = False  # Flag to track if quick_cut filename has been returned

    # Specify the directory to save the video files
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Initialize the last capture time
    last_capture_time = time.time()

    def process_frames():
        nonlocal recording, elapsed_time, quick_cut_saved, quick_cut_filename_returned, out, cap

        while recording:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the resulting frame
            cv2.imshow('Camera Feed', frame)

            # Capture frames at the specified interval
            current_time = time.time()
            if current_time - last_capture_time >= 1.0 / capture_frame_rate:
                # Write the frame to the video file
                out.write(frame)
                last_capture_time = current_time

                # Calculate mean intensity of the frame
                mean_intensity = calculate_mean_intensity(frame)
                intensity_data.append({'time': elapsed_time, 'mean_intensity': mean_intensity})

                # Check if the video has reached 90% settled
                if find_time and len(intensity_data) > 10:
                    time_data = np.array([entry['time'] for entry in intensity_data])
                    intensity_values = np.array([entry['mean_intensity'] for entry in intensity_data])

                    try:
                        # Fit the logistic growth model to the data
                        initial_guess = [np.max(intensity_values) - np.min(intensity_values), 
                                         np.min(intensity_values), 0.1, np.median(time_data)]
                        popt, _ = curve_fit(logistic_growth, time_data, intensity_values, p0=initial_guess, maxfev=2000)

                        # Calculate settling times
                        a, b, c, d = popt
                        settling_times = calculate_settling_times(a, b, c, d)

                        # Check if 90% settled time has been reached
                        if settling_times.get("90%") and elapsed_time >= settling_times["90%"]:
                            print(f"Video has reached 90% settled at {elapsed_time:.2f} seconds.")
                            if csv_path is not None:
                                with open(csv_path, 'a') as f:
                                    f.write(f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')},{elapsed_time:.2f}\n")
                            break
                    except RuntimeError as e:
                        print(f"Error fitting logistic growth model: {e}")

            # Save the recording after quick_cut duration
            if quick_cut is not None and elapsed_time >= quick_cut and not quick_cut_saved:
                print(f"Quick cut reached: {quick_cut} seconds. Saving recording...")
                out.release()
                recording = False
                out = None
                quick_cut_saved = True

                # Return video_filename after quick_cut is reached
                if not quick_cut_filename_returned:
                    quick_cut_filename_returned = True
                    print(f"Returning video filename after quick cut: {video_filename}")

        # Release resources when done
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    # Start recording
    recording = True
    start_time = time.time()
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = os.path.join(save_dir, f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    out = cv2.VideoWriter(video_filename, fourcc, capture_frame_rate, (width, height))
    print(f"Recording started: {video_filename}")

    # Start the frame processing in a separate thread
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.start()

    # Wait for the quick_cut time to pass
    while elapsed_time < quick_cut:
        elapsed_time = time.time() - start_time
        time.sleep(0.1)

    # Return the video filename after quick_cut
    print(f"Returning video filename after quick cut: {video_filename}")
    return video_filename

    # Wait for the processing thread to finish
    processing_thread.join()

    # Return elapsed time if find_time is enabled
    if find_time:
        return elapsed_time

    """
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Create a blank image for the timer and status
        status_img = 255 * np.ones((200, 400, 3), dtype=np.uint8)

        # Update the timer and status
        if recording:
            elapsed_time = time.time() - start_time
            status_text = f"Recording... {elapsed_time:.2f} sec"
            color = (0, 0, 255)  # Red color for recording
        else:
            status_text = "Not Recording"
            color = (0, 255, 0)  # Green color for not recording

        # Put the text on the status image
        cv2.putText(status_img, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Display the status image
        cv2.imshow('Status', status_img)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            # Start recording
            recording = True
            start_time = time.time()
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_filename = os.path.join(save_dir, f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
            out = cv2.VideoWriter(video_filename, fourcc, capture_frame_rate, (width, height))
            print(f"Recording started: {video_filename}")

        elif key == ord('s') and recording:
            # Stop recording
            recording = False
            out.release()
            out = None
            print("Recording stopped")

        if recording:
            # Capture frames at the specified interval
            current_time = time.time()
            if current_time - last_capture_time >= 1.0 / capture_frame_rate:
                # Write the frame to the video file
                out.write(frame)
                last_capture_time = current_time

            # Save the recording after quick_cut duration
            if quick_cut is not None and elapsed_time >= quick_cut and not quick_cut_saved:
                print(f"Quick cut reached: {quick_cut} seconds. Saving recording...")
                out.release()
                recording = False
                out = None
                quick_cut_saved = True
                if not find_time:
                    print("Recording stopped after quick cut")
                    break
                elif not quick_cut_filename_returned:
                    quick_cut_filename_returned = True
                    print(f"Returning video filename after quick cut: {video_filename}")
                    return video_filename
                    

            # Calculate mean intensity of the frame
            mean_intensity = calculate_mean_intensity(frame)
            intensity_data.append({'time': elapsed_time, 'mean_intensity': mean_intensity})

            # Check if the video has reached 90% settled
            if find_time and len(intensity_data) > 10:
                time_data = np.array([entry['time'] for entry in intensity_data])
                intensity_values = np.array([entry['mean_intensity'] for entry in intensity_data])

                try:
                    # Fit the logistic growth model to the data
                    initial_guess = [np.max(intensity_values) - np.min(intensity_values), 
                                     np.min(intensity_values), 0.1, np.median(time_data)]
                    popt, _ = curve_fit(logistic_growth, time_data, intensity_values, p0=initial_guess, maxfev=2000)

                    # Calculate settling times
                    a, b, c, d = popt
                    settling_times = calculate_settling_times(a, b, c, d)

                    # Check if 90% settled time has been reached
                    if settling_times.get("90%") and elapsed_time >= settling_times["90%"]:
                        print(f"Video has reached 90% settled at {elapsed_time:.2f} seconds.")
                        if csv_path is not None:
                            with open(csv_path, 'a') as f:
                                f.write(f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')},{elapsed_time:.2f}\n")
                        break
                except RuntimeError as e:
                    print(f"Error fitting logistic growth model: {e}")

        # Break the loop on 'q' key press
        if key == ord('q'):
            break

    # When everything is done, release the capture and video writer
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    if find_time:
        return elapsed_time
    else:
        return video_filename
    # Return the paths of the saved video and CSV file

    """

if __name__ == "__main__":
    run_camera()


"""
if recording:
            # Capture frames at the specified interval
            current_time = time.time()
            if current_time - last_capture_time >= 1.0 / capture_frame_rate:
                # Write the frame to the video file
                out.write(frame)
                last_capture_time = current_time
"""