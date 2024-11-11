#Importing
from keras.src.utils.image_dataset_utils import image_dataset_from_directory

#Specifying datasets
imageWidth, imageHeight = 224, 240
batchSize = 32
numberOfClasses = 4
dataDirectory = '../data/dog'

trainingDataset = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset = 'training',
    seed = 747,
    image_size = (imageHeight, imageWidth),
    batch_size = batchSize
)

validationDataset = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset = 'validation',
    seed = 747,
    image_size = (imageHeight, imageWidth),
    batch_size = batchSize
)

def getTrainingData():
    return trainingDataset

def getValidationData():
    return validationDataset