import random
import pathlib
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers


#take the directory of videos and splits them into training, validation, and test sets
#Args: 
#    directory: the directory of videos
#    regression_data: the regression data for each video
#    splits: Dictionary specifying the training, validation, test, etc.
#Return:
#    Mapping of the directories containing the subsections of data.
#    Mapping of the regression data for each video.

def split_videos(directory, regression_data, splits):
    #get the list of videos
    videos = os.listdir(directory)

    if len(videos) != len(regression_data):
        raise ValueError("The number of videos and regression data points must be the same.")

    #shuffle the videos and regression data together
    zipped = list(zip(videos, regression_data))
    random.shuffle(zipped)
    videos, regression_data = zip(*zipped)
    #create a dictionary to hold the splits
    split_dict = {}
    #create a dictionary to hold the regression data
    regression_dict = {}
    #create a counter to keep track of the current split
    counter = 0
    #loop through the splits
    for split in splits:
        #get the number of videos for the current split
        num_videos = splits[split]
        #get the videos for the current split
        split_videos = videos[counter:counter + num_videos]
        #add the videos to the dictionary
        split_dict[split] = split_videos
        #get the regression data for the current split
        split_regression_data = regression_data[counter:counter + num_videos]
        #add the regression data to the dictionary
        regression_dict[split] = split_regression_data
        #increment the counter
        counter += num_videos

    #return the dictionaries
    return split_dict, regression_dict


def format_frames(frame, output_size, crop_coordinates=None):
  #1920x1080
  #720x720
  #edges are 600 on left and right, and 180 on top and bottom
  #crop the image to 720x720
  #adjust position by 80 left and 180 down
  #start crop at 520, 360
  """
  Pad and resize an image from a video, with optional off-center cropping.

  Args:
    frame: Image that needs to be resized and padded.
    output_size: Pixel size of the output frame image.
    crop_coordinates: Tuple (offset_height, offset_width, target_height, target_width)
                      specifying the region to crop. If None, the center is cropped.

  Return:
    Formatted frame with padding of specified output size.
  """
  if frame is None:
        print("Error: Received a None frame in format_frames")
        return None
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame_height, frame_width, _ = frame.shape

  # If the frame is 1920x1080, change to 1280x720
  if frame_height == 1080 and frame_width == 1920:
    frame = tf.image.resize(frame, (720, 1280))
    frame_height, frame_width, _ = frame.shape

  if crop_coordinates:
    offset_height, offset_width = crop_coordinates
    target_height, target_width = output_size
    if offset_height + target_height > frame_height or offset_width + target_width > frame_width:
            raise ValueError(
                f"Cropping coordinates exceed frame dimensions. "
                f"Frame size: ({frame_height}, {frame_width}), "
                f"Crop: (offset_height={offset_height}, offset_width={offset_width}, "
                f"target_height={target_height}, target_width={target_width})"
            )
    frame = tf.image.crop_to_bounding_box(frame, offset_height, offset_width, target_height, target_width)
  else:
    frame = tf.image.resize_with_crop_or_pad(frame, *output_size)
  frame = tf.image.resize(frame, (224, 224))
  return frame


def frames_from_video_file(video_path, n_frames, output_size = (484,484), frame_step = 3, crop_coordinates = (190, 315)):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  
  if not src.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

  #video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
  #print(f"Video length of {video_path}: {video_length} frames")

  #need_length = 1 + (n_frames - 1) * frame_step
  #print(f"Need length: {need_length} frames")

  #if need_length > video_length:
  #  start = 0
  #else:
  #  if max_start<20:
  #    max_start = video_length - need_length
  #  else:
  #    max_start = 20
  #  start = random.randint(0, max_start + 1)
  #print(f"Starting frame: {start}")

  src.set(cv2.CAP_PROP_POS_FRAMES, 10)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  if not ret:
    print(f"Error1: Could not read frame from video file {video_path}")
    return None
  result.append(format_frames(frame, output_size, crop_coordinates))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size, crop_coordinates)
      result.append(frame)
    else:
      print(f"Error2: Could not read frame from video file {video_path}")
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result


class FrameGenerator:
  def __init__(self, path, regression_splits=None, n_frames=20, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: List of video file paths.
        regression_splits: Regression values for each video file.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.regression = regression_splits
    #self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    #self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def singleton(self):
    """Returns a set of frames for one video file."""
    video_frames = frames_from_video_file(self.path[0], self.n_frames)
    if video_frames is None:
      raise ValueError("Failed to extract frames from the video file.")
    
    video_frames = tf.convert_to_tensor(video_frames, dtype=tf.float32)

    regression_value = tf.convert_to_tensor(0, dtype=tf.int16)

    yield video_frames, regression_value
  
  def pairs(self):
    """Returns a set of frames with their associated label."""
    pairs = list(zip(self.path, self.regression))

    if self.training:
      random.shuffle(pairs)

    for path, regression_value in pairs:
      if path.suffix not in ['.mp4', '.avi', '.mov']:  # Add other supported video formats if needed
        print(f"Skipping non-video file: {path}")
        continue
      #print(f"Processing video file: {path.resolve()}")
      video_frames = frames_from_video_file(path, self.n_frames) 
      if video_frames is not None:
        yield video_frames, regression_value
      else:
        print(f"Failed to extract frames from {path}")

  def __call__(self):
    if isinstance(self.path, (list, tuple)):
      if len(self.path) == 1:
        return self.singleton()
      else:
        return self.pairs()
    elif isinstance(self.path, (str, pathlib.Path)):
      self.path = [self.path]
      return self.singleton()
    else:
      raise TypeError(f"Unsupported type for self.path: {type(self.path)}")
    


class PrintLayer(layers.Layer):
    def call(self, inputs):
        print(f"Shape of data before Flatten layer: {inputs.shape}")
        return inputs