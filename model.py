import random
import pathlib
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.callbacks import EarlyStopping
from utils.utils import split_videos
from utils.utils import FrameGenerator


regression_data1 =  [273.25, 173.52, 203.31, 227.25, 277.69, 213.21, 299.44, 360.26, 319.59, 372.49]
regression_data2 = [84.79, 71.87, 74.87, 77.27, 72.32, 76.91, 77.37, 85.43, 83.32, 83.50]
regression_data3 = [273.25, 84.79, 173.52, 71.87, 203.31, 74.87, 227.25, 77.27, 277.69, 72.32, 213.21, 76.91, 299.44, 77.37, 360.26, 85.43, 319.59, 83.32, 372.49, 83.50]
regression_data4 = [234.3158992, 238.3762008, 223.3395508 ,258.2964891, 308.3809155, 237.833691, 269.6809362, 240.1028949, 238.0344646, 289.8085463, 296.7480347, 254.8408521, 280.8549995, 305.4960162, 304.5373716, 332.1339075, 91.83076272, 102.9290284, 107.2722505, 118.4947244, 125.5493955, 143.4469968, 167.6710487, 168.928053, 161.7154215, 187.2395374, 208.7778182, 168.3711367, 209.7009893, 189.5577842, 209.69769, 198.7802618, 205.9488195, 209.9530789, 219.0467102, 221.408286, 221.3151488, 225.2788691, 230.7125019, 239.5862706, 236.1916802, 225.7250247, 240.7507197, 43.47757415, 40.49157473, 33.28122335, 32.0989265, 32.18500263, 22.77658, 22.39606886, 20.41376027, 22.6924993, 21.53471254, 23.88203075, 23.93024395, 21.85024589, 23.41806801, 28.39045941, 25.1815792, 26.33353658, 25.69546504, 24.17679749, 23.21302003]
data1 = "C:/Users/elika/Senior Design/Data/FeCl3-Videos-2_21_25/"
data2 = "C:/Users/elika/Senior Design/Data/Polymer-Videos-2_21_25"
data3 = "C:/Users/elika/Senior Design/Data/Combined-Videos-2_21_25"
data4 = "C:/Users/elika/Senior Design/Data/3_07-3_21-Videos"
save_path = "C:/Users/elika/Senior Design/Results/3_07-3_21-Videos_model/"

HEIGHT = 224
WIDTH = 224

data = data4  # Choose the appropriate data directory for your use case
regression_data = regression_data4  # Choose the appropriate regression data for your use case

#print files in data
print(f"Files in {data}:")
for file in os.listdir(data):
    print(file)

#get regression_data length
print(f"Length of regression_data: {len(regression_data)}")
data_size = len(regression_data)
train_split = int(0.6 * data_size)
test_split = int(0.2 * data_size)
val_split = data_size - train_split - test_split
print(f"Train split: {train_split}")
print(f"Validation split: {val_split}")
print(f"Test split: {test_split}")

subset_paths, regression_splits = split_videos(  
                        directory = data,
                        regression_data = regression_data,
                        splits = {"train": train_split, "val": val_split, "test": test_split})

#print(f"Subset paths: {subset_paths}")
#print(f"Regression splits: {regression_splits}")

n_frames = 20
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

print(f"Number of batches in train_ds: {len(list(train_ds))}")
print(f"Number of batches in val_ds: {len(list(val_ds))}")
print(f"Number of batches in test_ds: {len(list(test_ds))}")

# Define the model
model = models.Sequential()

# Add layers to the model
model.add(layers.Input(shape=(n_frames, HEIGHT, WIDTH, 3)))
model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling3D((1, 2, 2)))
model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
model.add(layers.Reshape((n_frames * 56 * 56 * 64,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='softplus'))

# Compile the model
frames, label = next(iter(train_ds))
model.build(frames)

train_ds = train_ds.repeat()  # Repeat the dataset for training
steps_per_epoch = len(subset_paths['train']) // batch_size

val_ds = val_ds.repeat()  # Repeat the validation dataset
val_steps = len(subset_paths['val']) // batch_size

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mae'])

history = model.fit(
    x=train_ds, 
    epochs=50, 
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[early_stopping])

# Save the model
model.save(save_path + 'model/video_model.h5')

# Save log file
log_file = save_path + 'training_log.txt'
with open(log_file, 'w') as f:
    f.write(f"Training history: {history.history}\n")
    f.write(f"Train steps per epoch: {steps_per_epoch}\n")
    f.write(f"Validation steps per epoch: {val_steps}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Number of frames: {n_frames}\n")
    f.write(f"Input shape: {(n_frames, HEIGHT, WIDTH, 3)}\n")

# Save the model architecture
model_json = model.to_json()
with open(save_path + 'model/model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
# Save the model weights
model.save_weights(save_path + 'model/model_weights.h5') 

#save test_ds
test_ds = test_ds.unbatch()
test_ds = test_ds.map(lambda x, y: y)
test_ds = np.array(list(test_ds.map(lambda x: x.numpy())))
np.save(save_path + 'model/test_ds.npy', test_ds)

# Save model splits to csv
with open(save_path + 'model/model_splits.csv', 'w') as f:
    f.write("train_paths,train,val_paths,val,test_paths,test\n")
    for i in range(len(subset_paths['train'])):
        f.write(f"{subset_paths['train'][i]},{regression_splits['train'][i]},{subset_paths['val'][i]},{regression_splits['val'][i]},{subset_paths['test'][i]},{regression_splits['test'][i]}\n")

