# Image-Classification
The following codes are designed for use with open source imagery processed with either QGIS or ArcGIS Pro. They were created for Planetary Remote Sensing. Each takes a TIF file and a CSV of ground truth sites and splits the image into training and testing sites and labels.

cnn.py is a file that utilizes the Keras model within the Tensorflow package to create a convolutional neural network based off the training and testing data, and outputs a model representing its accuracy. 
