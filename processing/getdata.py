#Importing
from processing.dataaugmentation import trainDataGenerator
from processing.dataaugmentation import validDataGenerator
from config import imageWidth
from config import imageHeight
from config import dataDirectory
from config import batchSize

trainingDataset = trainDataGenerator.flow_from_directory(
    dataDirectory,
    subset = 'training',
    class_mode = 'categorical',
    target_size=(imageWidth, imageHeight),
    batch_size = batchSize,
    shuffle = True,
    seed = 747,
)

validationDataset = validDataGenerator.flow_from_directory(
    dataDirectory,
    subset = 'validation',
    class_mode = 'categorical',
    target_size=(imageWidth, imageHeight),
    batch_size = batchSize,
    shuffle = True,
    seed = 747,
)

def getTrainingData():
    return trainingDataset

def getValidationData():
    return validationDataset