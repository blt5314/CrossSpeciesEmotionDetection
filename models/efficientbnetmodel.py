#Importing
from keras.src.applications.efficientnet import EfficientNetB0, EfficientNetB7
from processing.getdata import imageWidth
from processing.getdata import imageHeight
from processing.dataprocessing import getProcessedTrainingData
from processing.dataprocessing import getProcessedValidationData
from plotting.plotmodelhistory import plotHistory
from processing.getdata import numberOfClasses

#Retrieving data
trainingDataset = getProcessedTrainingData()
validationDataset = getProcessedValidationData()

#Specifying model
model = EfficientNetB0(
    include_top = True,
    weights = None,   #Try with None
    classes = numberOfClasses,
    input_shape = (imageHeight, imageWidth, 3),
)

#Creating model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=5, validation_data=validationDataset)

#Show results
plotHistory(hist)

#Save model
model.save('dog_effecientbnet0_multiclass_model.keras')