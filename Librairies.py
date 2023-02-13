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
