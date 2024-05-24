# Collin Campagna, 2024
# This script takes a raster dataset and a ground truth file to train a random forest classifier.
# use this to install necessary packages:
# py -m pip install rasterio pandas numpy matplotlib scikit-learn
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Machine learning models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# file inputs
# =============================================================================
# Open the dataset

# tif_file = input("Enter the path to the raster dataset: ")
# with rasterio.open(tif_file) as src:
with rasterio.open(r"Z:\TrainingData\Building\Streets.tif") as src:
    modis_data = src.read()
print(modis_data.shape)
# returns no. of bands, rows, columns

# Load ground truth data
# training_site = input("Enter the path to the training site: ")
# ground_truth = pd.read_csv(training_site)
ground_truth = pd.read_csv(r"Z:\TrainingData\Building\UrbanTrainingSites.csv")
print(ground_truth.head())

# # Get coordinates from the tfw file
# tfw_file = input("Enter the path to the tfw file: ")
# tfw_file = (r"Z:\TrainingData\Building\Streets.tfw")
# with open(tfw_file, 'r') as extent:
#     lines = extent.readlines()
#     x = float(lines[4])
#     y = float(lines[5])
#     print(x, y)
# =============================================================================

# Make sure the number of rows in the dataset and the ground truth file are the same
if len(modis_data) > len(ground_truth['Classname']):
    modis_data = modis_data[:len(ground_truth['Classname'])]
else:
    ground_truth = ground_truth[:len(modis_data)]


# Machine learning components
# =============================================================================
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(modis_data, ground_truth['Classname'], test_size=0.3, random_state=42)

# Reshape the data to 2D array
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# =============================================================================
# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=4)
clf.fit(X_train, y_train)

# Train SVM classifier. dont forget to toggle which classifier you want to use!
# clf = SVC(kernel='linear', C=1.0, random_state=0, probability=True)
# clf.fit(X_train, y_train)

# =============================================================================

# Predict the land cover type for the testing data
y_pred = clf.predict(X_test)
print(y_pred)

history = clf.fit(X_train, X_test, validation_data=(X_train, X_test))

# =============================================================================
# Evaluate the model:
test_loss, test_acc = y_pred.evaluate(X_train, X_test, verbose=2)
print('\nTest accuracy:', test_acc)


# =============================================================================
# Plot the training and validation loss and accuracy
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
# =============================================================================