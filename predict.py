import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import models
from utils.utils import FrameGenerator


model_path = "C:/Users/elika/Senior Design/Results/Clay_FeCl_Combined_model/"
model = models.load_model(model_path + "model/video_model.keras")
model_splits = pd.read_csv(model_path + 'model/model_splits.csv')
model_splits = model_splits.drop(model_splits.columns[:4], axis=1)
model_splits = model_splits.dropna(axis=0, how='all')
video_paths = model_splits['test_paths'].tolist()
#video_paths = [p for p in video_paths if isinstance(p, str) and p.strip()]
regression_data = model_splits['test'].tolist()
#regression_data = [r for r in regression_data if isinstance(r, (int, float))]  # Filter out non-numeric values

#print(f"Video paths: {video_paths}")
#print(f"Regression data: {regression_data}")

n_frames = 20
batch_size = 8
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
#load test set video data

#convert video paths to pathlib objects
video_paths = [pathlib.Path(p) for p in video_paths]

test_ds = tf.data.Dataset.from_generator(FrameGenerator(video_paths, regression_data, n_frames), 
                                          output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

"""
test_labels = np.load(model_path + 'model/test_ds.npy')
test_ds = tf.data.Dataset.from_tensor_slices(test_labels)
test_ds = test_ds.batch(8)  # Adjust batch size as needed
video_path = "C:/Users/elika/Senior Design/Data/3_07-3_21-Videos"
video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
video_files.sort()  # Sort the video files for consistent ordering
"""

predictions = model.predict(test_ds)
test_loss, test_mae = model.evaluate(test_ds, verbose=2)

# Convert predictions to numpy array
predictions = np.array(predictions)

#Calculate rmse
def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))

print(f"RMSE: {rmse(predictions, regression_data)}")

#calculate the average margin of error
moe = 0
for i in range(len(predictions)):
    moe += abs(predictions[i] - regression_data[i])/regression_data[i]
moe = moe/len(predictions)
print(f"Average % error: {moe}")

#print mae
print(f"MAE: {test_mae}")

#parity plot
#plot x=y line
#plt.figure(figsize=(10, 10))
sns.set_theme("poster")
plt.plot([0, 400], [0, 400], color='red', linewidth=3)
#plt.rcParams.update({'lines.markersize': 10})
plt.scatter(regression_data, predictions.flatten(), linewidths=1, edgecolors='black', zorder=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
#plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.title('Parity Plot')
#plt.xlim([25, 70])
#plt.ylim([25, 70])
plt.subplots_adjust(bottom=0.2)

# Ensure predictions is flattened and matches the length of regression_splits['test']
predictions_flattened = predictions.flatten()
if len(regression_data) != len(predictions_flattened):
	raise ValueError(f"Mismatch in lengths: regression_data has length {len(regression_data)}, "
					 f"but predictions has length {len(predictions_flattened)}")

# Calculate and show r^2
r2 = np.corrcoef(regression_data, predictions_flattened)[0, 1]
plt.text(180, 50, f"R^2: {r2:.2f}", fontsize=24)

plt.savefig(model_path + 'model/parity_plot.png', dpi=300)

#save predictions to csv along with video paths
predictions = predictions.flatten()
predictions_df = pd.DataFrame(predictions, columns=['Predictions']) 
predictions_df['True Values'] = regression_data
predictions_df['Video Path'] = video_paths
#include rmse, moe, mae, and r2 in the csv while filling the rest with NaN
predictions_df['RMSE'] = rmse(predictions, regression_data)
predictions_df['MAE'] = test_mae
predictions_df['MOE'] = [moe] * len(predictions_df)
predictions_df['R2'] = [r2] * len(predictions_df)

predictions_df.to_csv(model_path + 'model/predictions.csv', index=False)


