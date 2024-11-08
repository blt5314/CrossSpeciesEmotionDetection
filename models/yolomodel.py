#Importing
import numpy as np
import tensorflow as tf
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.applications.efficientnet_v2 import EfficientNetV2
from keras.src.models import Model
from keras.src.models import Sequential
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from keras.src import layers
from keras.src.losses.losses import SparseCategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt