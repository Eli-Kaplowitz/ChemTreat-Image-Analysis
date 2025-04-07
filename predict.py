import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import models


model_path = "C:/Users/elika/Senior Design/Results/3_07-3_21-Videos_model/"
model = models.load_model(model_path + "model/model.h5")
model_splits = np.load(model_path + 'model/model_splits.npy', allow_pickle=True).item()


test_labels = np.load(model_path + 'model/test_ds.npy')
test_ds = tf.data.Dataset.from_tensor_slices(test_labels)
test_ds = test_ds.batch(8)  # Adjust batch size as needed
video_path = "C:/Users/elika/Senior Design/Data/3_07-3_21-Videos"
video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
video_files.sort()  # Sort the video files for consistent ordering

predictions = model.predict(test_ds)

# Convert predictions to numpy array
predictions = np.array(predictions)

#Calculate rmse
def rmse(predictions, targets):
    return np.sqrt(np.mean(((predictions - targets) ** 2)))

print(f"RMSE: {rmse(predictions, test_labels)}")

#calculate the average margin of error
moe = 0
for i in range(len(predictions)):
    moe += abs(predictions[i] - test_labels[i])/test_labels[i]
moe = moe/len(predictions)
print(moe)

#parity plot
#plot x=y line
sns.set_theme("poster")
plt.plot([0, 400], [0, 400], color='red', linewidth=3)
#plt.rcParams.update({'lines.markersize': 10})
plt.scatter(regression_splits['test'], predictions.flatten(), linewidths=1, edgecolors='black', zorder=2)
plt.xlabel('True Values (sec)')
plt.ylabel('Predictions (sec)')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.title('Parity Plot')

# Ensure predictions is flattened and matches the length of regression_splits['test']
predictions_flattened = predictions.flatten()
if len(regression_splits['test']) != len(predictions_flattened):
	raise ValueError(f"Mismatch in lengths: regression_splits['test'] has length {len(regression_splits['test'])}, "
					 f"but predictions has length {len(predictions_flattened)}")

# Calculate and show r^2
r2 = np.corrcoef(regression_splits['test'], predictions_flattened)[0, 1]
plt.text(180, 50, f"R^2: {r2:.2f}", fontsize=24)

plt.show()