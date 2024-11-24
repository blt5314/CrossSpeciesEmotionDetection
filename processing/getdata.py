#Importing
from processing.dataaugmentation import trainDataGenerator
from processing.dataaugmentation import validDataGenerator

#Specifying datasets
imageWidth, imageHeight = 224, 224
numberOfClasses = 5
dataDirectory = '../data/dog'
batchSize = 32

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