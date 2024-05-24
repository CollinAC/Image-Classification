# =============================================================================
# Collin Campagna, 2024
# BCIT GIST 8125
# Faculty Supervisor: Joshua Macdougall
# Industry Sponsor: Ramin Azar, Planetary Remote Sensing.
# This script takes a raster dataset and a ground truth file created in ArcGIS Pro to train a convolutional neural network.
# The Raster image in this application is open source aerial imagery of Port Coquitlam, BC.
# The ground truth file is a csv file created in ArcGIS Pro that contains the class codes for the training data.
# The CSV file contains the class codes for the training data. This serves as the primary key to join to the raster dataset.
# The model is trained with the training data and then tested with the test data.
# The model is then evaluated and the training and validation accuracy and loss are plotted.
                                ### IMPORTANT! ###          
# Before running the script, ensure that Python is installed on the machine.
# Run this in terminal before use in order to have all the necessary packages installed:
# py -m pip install numpy tensorflow keras matplotlib scikit-learn
# =============================================================================

import rasterio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set GPU memory growth to avoid allocating all GPU memory upfront
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
# =============================================================================
# Load the data
# =============================================================================

# Load the aerial image data as a TIFF file. Re-path as necessary.
with rasterio.open(r"Z:\TrainingData\Building\Streets.tif") as src:
    band1 = src.read(1)
    band2 = src.read(2)
    band3 = src.read(3)
image_data = np.dstack((band1, band2, band3))

# Load the training site data. Re-path as necessary.
ground_truth = pd.read_csv(r"Z:\TrainingData\Building\UrbanTrainingSites.csv")

# =============================================================================
# Pre-process the data
# =============================================================================

# Set the land cover labels to Classcode as a float
land_cover_labels = ground_truth['Classcode'].astype(float)

# make the number of rows in the dataset and the ground truth file the same
if len(image_data) > len(land_cover_labels):
    data = image_data[:len(land_cover_labels)]
else:
    land_cover_labels = land_cover_labels[:len(image_data)]
    data = image_data[:len(land_cover_labels)]

# Normalize pixel values to be between 0 and 1
data = data / 255.0

# Get the dimensions of the image data
image_dimensions = image_data.shape
image_rows = image_dimensions[0]
image_columns = image_dimensions[1]
image_bands = image_dimensions[2]
print(f"The dimensions of the image are: {str(image_dimensions)}")

# Reshape the data to 4D array
image_data = image_data.reshape(-1, image_rows, image_columns, image_bands)  # -1 means the number of rows is inferred
input_shape = image_data.shape
print(f"The shape going into the model is {str(input_shape)}")
# =============================================================================
# Train the model
# =============================================================================

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(data, land_cover_labels, test_size=0.2, random_state=42)

# Create the model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(13362, 21361, 3)),   # this number should reflect the rows, columns, and bands in the image
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(5)  # 5 classes in training data
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=1, epochs=10, verbose=1)

# =============================================================================
# Model evaluation
# =============================================================================

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

#==============================================================================
# Plot the training and validation accuracy
# =============================================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

print('End of script')