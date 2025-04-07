import cv2
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import models

"""
This script is used to perform live predictions on a video stream using a pre-trained model.
It captures video frames, processes them, and uses the model to predict the output.
It also includes functions for calculating settling times, and plotting fitted curves.
The script is designed to work with a specific video format and model architecture.
"""

