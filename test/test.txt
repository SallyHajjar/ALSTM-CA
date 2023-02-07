!pip install rasterio
!pip install pyrsgis
import os
import numpy as np
import numpy
import gdal
import cv2
import sys
import random
import glob
import time
import cv2
import math
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pandas as pd
from random import randint, shuffle, choice
from imutils import paths
import argparse
from osgeo import gdal
from copy import deepcopy
from PIL import Image
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import itertools
from collections import Counter
import rasterio as rio
from rasterio.plot import show
from pyrsgis import raster
from sklearn.ensemble import RandomForestClassifier
from pyrsgis.convert import changeDimension, array_to_table

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, Layer, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

import keras.backend as K
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
def rst(file):
    Raster = gdal.Open(file)
    Band=Raster.GetRasterBand(1)
    Array=Band.ReadAsArray()
    return(Array)

def create_mask(radius, omit_center):
	mask_size = 2*radius+1
	mask = np.ones((mask_size, mask_size))
	if omit_center:
		mask_center = radius
		mask[mask_center, mask_center] = np.nan
	return mask

def find_neighbors(p_arr, radius, row_number, column_number):
	return [[p_arr[i][j] if  i >= 0 and i < len(p_arr) and j >= 0 and j < len(p_arr[0]) else np.nan
		for j in range(column_number-radius, column_number+radius+1)]
		for i in range(row_number-radius, row_number+radius+1)]

def calc_neighbors(p_original_arr):
	neighbors_radius = 3
	nb_rows = p_original_arr.shape[0]
	nb_cols = p_original_arr.shape[1]
	result = np.zeros(shape=(nb_rows, nb_cols))
	mask = create_mask(neighbors_radius, True)
	for i in range(0, nb_rows):
		for j in range(0, nb_cols):
			neighbors_array = np.array(find_neighbors(p_original_arr, neighbors_radius, i, j))
			cell_res = neighbors_array*mask
			sum = np.nansum(cell_res)
			count = np.count_nonzero(~np.isnan(cell_res))
			cell_val = sum/count if count > 0 else 0
			result[i, j] = cell_val
			
	return result

def calc_neighbors_c(p_original_arr, number_classes):
	neighbors_radius = 3
	nb_rows = p_original_arr.shape[0]
	nb_cols = p_original_arr.shape[1]
	results = np.zeros(shape=(number_classes, nb_rows, nb_cols))
	mask = create_mask(neighbors_radius, True)
	for i in range(0, nb_rows):
		for j in range(0, nb_cols):
			neighbors_array = np.array(find_neighbors(p_original_arr, neighbors_radius, i, j))
			cell_res = neighbors_array*mask
			count = np.count_nonzero(~np.isnan(cell_res))
			sums = np.zeros(number_classes)
			for a in range(0,cell_res.shape[0]):
				for b in range(0,cell_res.shape[1]):
					if cell_res[a,b] is not None and not np.isnan(cell_res[a,b]):
						cell_val = int(cell_res[a,b])
						sums[cell_val] = sums[cell_val] + 1

			for k in range(0, 4):
				results[k, i, j] = sums[k]/count if count > 0 else 0
			
	return results

def convertToClasses(arr):
	arr_classes=arr
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]): 
			if arr[i,j] >= 0  and arr[i,j]  <= 24:
				arr_classes[i,j]=0
			elif arr[i,j] >= 25  and arr[i,j]  <= 102:
				arr_classes[i,j]=1
			elif arr[i,j] >= 103  and arr[i,j]  <= 499:
				arr_classes[i,j]=2
			elif arr[i,j] >= 500  and arr[i,j]  <= 2500:
				arr_classes[i,j]=3
	return arr_classes

def convertToOneColumnArray(arr):
	arr = array_to_table(arr)
	arr = arr.reshape((arr.shape[0], 1))
	return arr

def computeTransitions(p_x, p_y):	
	a=[]
	for i in range(p_x.shape[0]):
		for j in range(p_x.shape[1]): 
			if p_x[i,j] != p_y[i,j]:
				a.append(abs(p_y[i,j] - p_x[i,j])+(p_x[i,j]+3)*10)
			else:
				a.append(p_x[i,j])
	a=np.reshape(a,(p_x.shape[0], p_x.shape[1]))
	return a

def convertModelOutputToLabels(outPred):
	return np.argmax(np.array(outPred), axis=-1)

def calculatePlotConfusionMatrix(yTrue, yPred, yPrev):
	classes = np.unique(yTrue)
	target_names = ["Class {}".format(i) for i in classes]
	cMatrix = confusion_matrix(yTrue, yPred)
	classificationReport = classification_report(yTrue, yPred, target_names=target_names)
	accuracyScore = accuracy_score(yTrue, yPred)
	precisionScore = precision_score(yTrue, yPred, average='macro') #weighted
	recallScore = recall_score(yTrue, yPred, average='macro')

	A=0
	B=0
	C=0
	D=0
	for i in range(len(yPred)):
		if yPrev[i] != yTrue[i]  and yPrev[i] == yPred[i]:
			A=A+1
		elif yPrev[i] != yTrue[i]  and yPrev[i] != yPred[i]:
			B=B+1
		if yTrue[i]  == yPred[i]:
			C=C+1
		elif yPrev[i] == yTrue[i]  and yPrev[i] != yPred[i]:
			D=D+1
	
	FoM = (B/(A+B+C+D))

	print("Confusion matrix:\n", cMatrix)
	print("Classification Report:\n", classificationReport)
	print("Accuracy:", accuracyScore)
	print("Precision Score : ", precisionScore)
	print("Recall Score : ", recallScore)
	print("FoM : ", FoM)
 
	# plot confusion matrix
	plt.imshow(cMatrix, interpolation='nearest', cmap=plt.cm.Greens)
	plt.title("Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = cMatrix.max() / 2.
	for i, j in itertools.product(range(cMatrix.shape[0]), range(cMatrix.shape[1])):
			plt.text(j, i, format(cMatrix[i, j], fmt),
							horizontalalignment="center",
							color="white" if cMatrix[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	return cMatrix, classificationReport, accuracyScore, precisionScore, recallScore

def calculateTransitionMatrix(yIn, yOut):
	nb_unique_classes = len(np.unique(yOut))
	transitionMatrix = np.zeros((nb_unique_classes, nb_unique_classes))

	for i in range(yOut.shape[0]):
		transitionMatrix[yIn[i], yOut[i]] = transitionMatrix[yIn[i], yOut[i]] + 1

	for i in range(transitionMatrix.shape[0]):
		for j in range(transitionMatrix.shape[1]):
			print("Class" + str(i) + " to " + str(j) + ": ", int(transitionMatrix[i, j]))
			
	return transitionMatrix

def plotMap(map):
	show(map)
 
def predict(model, xTest, yPrevious):
	yTestPredicted = model.predict(xTest)
	yTestPredicted_labels = convertModelOutputToLabels(yTestPredicted)
 
	for i in range(0, len(yTestPredicted_labels)):
			if yTestPredicted_labels[i] < yPrevious[i]:
				yTestPredicted_labels[i]= yPrevious[i]
	return yTestPredicted_labels

def analyze_prediction(yPred, yTrue, yPrevious, printTitle = None):
	if printTitle is not None:
		print(printTitle + ":\n")

	yTrue_t = array_to_table(yTrue)
	yPrevious_t = array_to_table(yPrevious)

	calculatePlotConfusionMatrix(yTrue_t, yPred, yPrevious_t)
	calculateTransitionMatrix(yPrevious_t, yPred)
featuresFolder = '/content/drive/MyDrive/SusDens/dataandcodes/features/'
outputFolder = '/content/drive/MyDrive/SusDens/output/'
_, dens_map_2000 = raster.read(featuresFolder + 'BU2000.tif', bands=1)
_, dens_map_2010 = raster.read(featuresFolder + 'BU2010.tif', bands=1)
_, dens_map_2020= raster.read(featuresFolder + 'BU2020.tif', bands=1)

dens_map_2000_c = convertToClasses(dens_map_2000)
dens_map_2010_c = convertToClasses(dens_map_2010)
dens_map_2020_c = convertToClasses(dens_map_2020)
xtrain = convertToOneColumnArray(dens_map_2000)
xtest = convertToOneColumnArray(dens_map_2010)

featuresArray_train = [xtrain]
featuresArray_test = [xtest]
for featureDict in featuresDict:
	if featureDict['enabled']:
		_, arr = raster.read(featuresFolder + featureDict['filename'], bands = featureDict['bands'])
		if featureDict['pixels_to_remove']['0'] is not None:
			arr = np.delete(arr, featureDict['pixels_to_remove']['0'], 0)
		if featureDict['pixels_to_remove']['1'] is not None:
			arr = np.delete(arr, featureDict['pixels_to_remove']['1'], 1)
		arr = convertToOneColumnArray(arr)
		featureDict['arr'] = arr
		featuresArray_train.append(arr)
		featuresArray_test.append(arr)
dens_map_2000_n = calc_neighbors_c(dens_map_2000_c, 4)
dens_map_2010_n = calc_neighbors_c(dens_map_2010_c, 4)

for i in range(4):
  featuresArray_train.append(convertToOneColumnArray(dens_map_2000_n[i]))
  featuresArray_test.append(convertToOneColumnArray(dens_map_2010_n[i]))
featuresArray_train = np.concatenate(np.array(featuresArray_train), 1)
featuresArray_test = np.concatenate(np.array(featuresArray_test), 1)
outputFolder = '/content/drive/MyDrive/SusDens/output/'

# print(featuresArray_train.shape)
# np.savetxt(outputFolder + "featuresArray_train.csv", featuresArray_train, delimiter=",")
# np.savetxt(outputFolder + "featuresArray_test.csv", featuresArray_test, delimiter=",")

X = convertToOneColumnArray(dens_map_2000_c)
y = array_to_table(dens_map_2010_c)

X_transitions = convertToOneColumnArray(computeTransitions(dens_map_2000_c, dens_map_2010_c))

featuresX = np.concatenate([featuresArray_train, X_transitions], 1)
xTrain, xValidation, yTrain, yValidation = train_test_split(featuresX, y, test_size=0.2, random_state=22, stratify = X_transitions)
xTrain=np.delete(xTrain, xTrain.shape[1] - 1,1)
xValidation=np.delete(xValidation, xValidation.shape[1] - 1,1)
print(xTrain.shape, xValidation.shape, yTrain.shape, yValidation.shape)
# Normalise the data
xTrain = xTrain / 188914
xValidation = xValidation / 188914
xTest = featuresArray_test / 188914

# Reshape the data
xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1])) #Another additional pre-processing step is to reshape the features from two-dimensions to three-dimensions, such that each row represents an individual pixel.
xValidation = xValidation.reshape((xValidation.shape[0], 1, xValidation.shape[1]))
xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
from keras.layers import Layer
import keras.backend as K
class attention(Layer):
    def _init_(self,**kwargs):
        super(attention,self)._init_(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()
model = Sequential()
model.add(LSTM(56, input_shape=(1,21), return_sequences=True))
model.add(Dropout(0.3))
model.add(attention())
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(4))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
model.fit(xTrain, yTrain, batch_size=100, epochs=100,validation_data=(xTest, yTest), verbose=1)
yPred = predict(model, xTest, array_to_table(dens_map_2010_c))
np.savetxt(outputFolder + "yPred.csv", yPred, delimiter=",")
np.savetxt(outputFolder + "yIn.csv", array_to_table(dens_map_2010_c), delimiter=",")
analyze_prediction(yPred, dens_map_2020_c, dens_map_2010_c, 'Model')
plotMap(dens_map_2010_c[200:400, 300:588])
plotMap(dens_map_2020_c[200:400, 300:588])
plotMap(yPred.reshape(dens_map_2020_c.shape[0], dens_map_2020_c.shape[1])[200:400, 300:588])
