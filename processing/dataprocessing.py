from keras.src import layers
import tensorflow as tf
from processing.getdata import getTrainingData
from processing.getdata import getValidationData
from processing.getdata import numberOfClasses

#Specifying augmentation layers
augmentationLayers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]

#Augmentating image with augmentation layers
def imageAugmentation(image):
    for layer in augmentationLayers:
        image = layer(image)
    return image

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
trainingDataset = trainingDataset.prefetch(tf.data.AUTOTUNE)

validationDataset = validationDataset.map(testInputPreProcessing, num_parallel_calls=tf.data.AUTOTUNE)

def getProcessedTrainingData():
    return trainingDataset

def getProcessedValidationData():
    return validationDataset