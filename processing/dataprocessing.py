from keras import Sequential
from keras.src import layers
import tensorflow as tf
from getdata import getTrainingData
from getdata import getValidationData
from getdata import batchSize
from getdata import numberOfClasses

#Specifying augmentation layers
augmentationLayers = Sequential([
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
])

#Augmentating image with augmentation layers
def imageAugmentation(images):
    for layer in augmentationLayers:
        images = layer(images)
    return images

#Preprocessing for training data
def trainInputPreProcessing(image, label):
    image = imageAugmentation(image)
    label = tf.one_hot(label, numberOfClasses)
    return image, label

#Preprocessing for test/validation data
def testInputPreProcessing(image, label):
    label = tf.one_hot(label, numberOfClasses)
    return image, label

trainingDataset = getTrainingData()
validationDataset = getValidationData()

trainingDataset = trainingDataset.map(trainInputPreProcessing, num_parallel_calls=tf.data.AUTOTUNE)
trainingDataset = trainingDataset.batch(batch_size=batchSize, drop_remainder=True)
trainingDataset = trainingDataset.prefetch(tf.data.AUTOTUNE)

validationDataset = validationDataset.map(testInputPreProcessing, num_parallel_calls=tf.data.AUTOTUNE)
validationDataset = validationDataset.batch(batch_size=batchSize, drop_remainder=True)

def getProcessedTrainingData():
    return trainingDataset

def getProcessedValidationData():
    return validationDataset