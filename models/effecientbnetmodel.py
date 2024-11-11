#Importing
from keras.src.applications.efficientnet import EfficientNetB0
from processing.getdata import imageWidth
from processing.getdata import imageHeight
from processing.dataprocessing import getProcessedTrainingData
from processing.dataprocessing import getProcessedValidationData
from plotting.plotmodelhistory import plotHistory

#Retrieving data
trainingDataset = getProcessedTrainingData()
validationDataset = getProcessedValidationData()

#Specifying model
model = EfficientNetB0(
    include_top=False,
    weights='imagenet',   #Try with None
    classes=4,
    input_shape=(imageHeight, imageWidth, 3),
)

#Creating model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=10, validation_data=validationDataset)

#Show results
plotHistory(hist)
