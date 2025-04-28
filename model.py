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
regression_data25 = [296.3947639863612, 59.29759116577222, 271.1137036859074, 35.00814904403077, 277.3375119246293, 62.975402043388144, 289.8261088824782, 64.79454709, 292.9268792437999, 67.96623781845882, 306.6297419086934, 79.10460210498405, 298.59346218722703, 81.83847265, 310.0444128, 37.15570045218213, 279.87226272154675, 42.78107581226777, 281.99402868546935, 50.56634461772411, 272.139951, 67.51738465028785, 283.48626602588627, 53.953535177348165, 299.70191820923134, 60.076302673909204, 279.1338214667887, 57.204503865849986, 296.66436600677025, 58.66191148391129]
regression_data_Clay = [16.907536804838152, 29.890605632166388, 36.46199004157364, 39.76297151, 43.44187582106522, 41.97090754261376, 45.26913482863291, 44.17612496380685, 47.05333742901979, 47.06515884750002, 48.04142050096023, 50.554896835361376, 51.09661754050004, 50.69072567282765, 51.86840039573379, 52.41052419490371, 52.624615990225934, 54.26106784608635, 51.860255251399686, 53.39755459636431, 55.611369554861085, 55.11024615648594, 55.59111019584038, 55.443011686335076, 20.54461165387771, 35.980289088017855, 37.790280407582195, 35.74556687, 37.21743426389379, 40.27400253842077, 40.328808243993855, 39.31509018088954, 41.17213384499064, 44.25913561, 42.17933299853956, 41.16902864769375, 44.47698703655051, 44.68774198410488, 45.64368688045422, 45.291699013182715, 44.55552454894576, 45.12796843803846, 44.11998737676535, 46.561830671649375, 45.61772214766259, 46.034483959435626, 46.222091740562476, 45.027061568420265, 46.84168174293055, 43.97946641439015, 46.11738808139695, 44.885693613444204, 19.955295460276005, 5.633392910921415, 30.195271817730436, 51.65225653593505, 61.58284007635173, 60.00883298676159, 64.76601503478025, 71.62888474057046]
regression_data5 = regression_data4 + regression_data25
regression_data6 = regression_data4 + regression_data_Clay + regression_data25
data1 = "C:/Users/elika/Senior Design/Data/FeCl3-Videos-2_21_25/"
data2 = "C:/Users/elika/Senior Design/Data/Polymer-Videos-2_21_25"
data3 = "C:/Users/elika/Senior Design/Data/Combined-Videos-2_21_25"
data4 = "C:/Users/elika/Senior Design/Data/3_07-3_21-Videos"
data5 = "C:/Users/elika/Senior Design/Data/3_07-3_21-3_25-Videos"
data6 = "C:/Users/elika/Senior Design/Data/Clay_FeCl_Combined"
save_path = "C:/Users/elika/Senior Design/Results/3_07-3_21-3_25-Videos_model2/"

HEIGHT = 224
WIDTH = 224

data = data6  # Choose the appropriate data directory for your use case
regression_data = regression_data6  # Choose the appropriate regression data for your use case

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

#check if save_path exists and create it if it doesn't
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path + 'model')

# Save the model
model.save(save_path + 'model/video_model.keras')

# Save log file
log_file = save_path + 'training_log.txt'
with open(log_file, 'w') as f:
    f.write(f"Training history: {history.history}\n")
    f.write(f"Train steps per epoch: {steps_per_epoch}\n")
    f.write(f"Validation steps per epoch: {val_steps}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Number of frames: {n_frames}\n")
    f.write(f"Input shape: {(n_frames, HEIGHT, WIDTH, 3)}\n")

"""
# Save the model architecture
model_json = model.to_json()
with open(save_path + 'model/model_architecture.json', 'w') as json_file:
    json_file.write(model_json)
# Save the model weights
model.save_weights(save_path + 'model/model_weights.keras') 
"""

# Save model splits to csv

max_length = max(len(subset_paths['train']),
                 len(regression_splits['train']),
                 len(subset_paths['val']),
                 len(regression_splits['val']),
                 len(subset_paths['test']),
                 len(regression_splits['test']))

with open(save_path + 'model/model_splits.csv', 'w') as f:
    f.write("train_paths,train,val_paths,val,test_paths,test\n")

    for i in range(max_length):
        train_path = subset_paths['train'][i] if i < len(subset_paths['train']) else ""
        train_split = regression_splits['train'][i] if i < len(regression_splits['train']) else ""
        val_path = subset_paths['val'][i] if i < len(subset_paths['val']) else ""
        val_split = regression_splits['val'][i] if i < len(regression_splits['val']) else ""
        test_path = subset_paths['test'][i] if i < len(subset_paths['test']) else ""
        test_split = regression_splits['test'][i] if i < len(regression_splits['test']) else ""

        f.write(f"{train_path},{train_split},{val_path},{val_split},{test_path},{test_split}\n")

#save test_ds
test_ds = test_ds.unbatch()
test_ds = test_ds.map(lambda x, y: y)
test_ds = np.array(list(test_ds.map(lambda x: x.numpy())))
np.save(save_path + 'model/test_ds.npy', test_ds)   