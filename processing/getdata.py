#Importing
from keras.src.utils.image_dataset_utils import image_dataset_from_directory

#Specifying datasets
imageWidth, imageHeight = 456, 456
numberOfClasses = 5
dataDirectory = '../data'
batchSize = 32

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
    labels = 'inferred',
    validation_split=0.2,
    subset = 'validation',
    shuffle = True,
    seed = 747,
    image_size = (imageHeight, imageWidth),
    batch_size = batchSize
)

def getTrainingData():
    return trainingDataset

def getValidationData():
    return validationDataset