# Importing necessary modules and variables
# Importing trainDataGenerator and validDataGenerator for data augmentation and loading datasets
from processing.dataaugmentation import trainDataGenerator
from processing.dataaugmentation import validDataGenerator
# Importing configuration variables for image dimensions, data directory, and batch size
from config import imageWidth
from config import imageHeight
from config import dataDirectory
from config import batchSize

# Generating the training dataset
trainingDataset = trainDataGenerator.flow_from_directory(
    dataDirectory,
    subset='training',
    class_mode='categorical',
    target_size=(imageWidth, imageHeight),
    batch_size=batchSize,
    shuffle=True,
    seed=747,
)

# Generating the validation dataset
validationDataset = validDataGenerator.flow_from_directory(
    dataDirectory,
    subset='validation',
    class_mode='categorical',
    target_size=(imageWidth, imageHeight),
    batch_size=batchSize,
    shuffle=True,
    seed=747,
)

# Function to retrieve the training dataset
def getTrainingData():
    return trainingDataset

# Function to retrieve the validation dataset
def getValidationData():
    return validationDataset
