import cv2
import pathlib
import numpy as np
import threading
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import models
from utils.utils import FrameGenerator
from Ethan_video_testing2.video_processing.intensity_analysis import calculate_mean_intensity
from Camera import run_camera

"""
This script is used to perform live predictions on a video stream using a pre-trained model.
It captures video frames, processes them, and uses the model to predict the output.
It also includes functions for calculating settling times, and plotting fitted curves.
The script is designed to work with a specific video format and model architecture.
"""

"""
Activate camera recording for one minute and 10 seconds and save the video to a file.
Run a modified version of FrameGenerator to process video frames
Make a prediction using the pre-trained model
Display the video as it's recorded, then display the prediction in a plot
Let the camera continue while the prediction is made until it reaches a point defined as 90% settled using mean intensity.
Stop the camera and compare the prediction to the actual data.
Save the prediction to a file.
"""

model_path = "C:/Users/elika/Senior Design/Results/3_07-3_21-3_25-Videos_model2/"
model = models.load_model(model_path + "model/video_model.keras")

save_dir = "C:/Users/elika/Senior Design/Results/live_predict/1"

video_path = os.path.join(save_dir, "live_video.mp4")
csv_path = os.path.join(save_dir, "settling_times.csv")

video_filename = run_camera(save_dir, 70, False, csv_path)

print(f"Video saved to: {video_filename}")

n_frames = 20
batch_size = 1

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32), 
                    tf.TensorSpec(shape = (), dtype = tf.int16))


video_filename = pathlib.Path(video_filename)

print("Video filename as pathlib")

test_ds = tf.data.Dataset.from_generator(FrameGenerator(video_filename, n_frames), 
                                          output_signature = output_signature)

print("test_ds created")

test_ds = test_ds.batch(batch_size)

print("test_ds batched")

prediction = model.predict(test_ds)

print(f"Prediction: {prediction}")
