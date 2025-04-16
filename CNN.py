import random
import pathlib
import cv2
import einops
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
from keras import layers
import os


def split_videos(directory, regression_data, splits):
    #get the list of videos
    videos = os.listdir(directory)
    #shuffle the videos
    random.shuffle(videos)
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
        split_videos = videos[counter:counter+num_videos]
        #add the videos to the dictionary
        split_dict[split] = split_videos
        #get the regression data for the current split
        split_regression_data = regression_data[counter:counter+num_videos]
        #add the regression data to the dictionary
        regression_dict[split] = split_regression_data
        #increment the counter
        counter += num_videos

    #return the dictionaries
    return split_dict, regression_dict

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  if frame is None:
        print("Error: Received a None frame in format_frames")
        return None
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
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

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
  #print(f"Video length of {video_path}: {video_length} frames")

  need_length = 1 + (n_frames - 1) * frame_step
  #print(f"Need length: {need_length} frames")

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)
  #print(f"Starting frame: {start}")

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  if not ret:
    print(f"Error: Could not read frame from video file {video_path}")
    return None
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      print(f"Error: Could not read frame from video file {video_path}")
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
  def __init__(self, path, regression_splits, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.regression = regression_splits
    #self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    #self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files(self):
    video_paths = self.path
    #print(f"Found {len(video_paths)} video files in {self.path}")
    return video_paths

  def __call__(self):
    video_paths = self.get_files()
    pairs = list(zip(video_paths, self.regression))

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

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos
  
class PrintLayer(layers.Layer):
    def call(self, inputs):
        print(f"Shape of data before Flatten layer: {inputs.shape}")
        return inputs

###########################################
regression_data =  [43, 52, 63, 59, 75, 66, 62, 70, 80, 102]
data = "C:/Users/elika/Senior Design/Data/FeCl3-Videos-2_21_25/"
HEIGHT = 224
WIDTH = 224
###########################################

subset_paths, regression_splits = split_videos(  
                        directory = data,
                        regression_data = regression_data,
                        splits = {"train": 6, "val": 2, "test": 2})

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

base_dir = pathlib.Path(data)

#convert subset paths to pathlib objects
#subset_paths = {key: pathlib.Path(value) for key, value in subset_paths.items()}
subset_paths = {key: [base_dir / pathlib.Path(p).name for p in value] for key, value in subset_paths.items()}

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], regression_splits['train'], n_frames, training=True),
                                          output_signature = output_signature)


# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], regression_splits['val'], n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], regression_splits['test'], n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

# Is this necessary?
#resize_video = ResizeVideo(HEIGHT, WIDTH)
#frames = resize_video(frames)

model = models.Sequential()
model.add(layers.Input(shape=(n_frames, HEIGHT, WIDTH, 3)))
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.Reshape((10 * 56 * 56 * 64,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='softplus'))
model.summary()

frames, label = next(iter(train_ds))
model.build(frames)

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mae'])

history = model.fit(x=train_ds, epochs=50, validation_data=val_ds)

#plot the training and validation loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 4000])
plt.legend(loc='lower right')
plt.show()

#plot the training and validation mae
plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label = 'val_mae')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.ylim([0, 60])
plt.legend(loc='lower right')
plt.show()

test_loss, test_mae = model.evaluate(test_ds, verbose=2)
print(test_loss, test_mae)