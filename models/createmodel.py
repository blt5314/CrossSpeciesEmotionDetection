#Importing
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Dropout, Flatten, BatchNormalization, Dense, Activation, GlobalAveragePooling2D
from processing.getdata import imageWidth
from processing.getdata import imageHeight
from processing.getdata import getTrainingData
from processing.getdata import getValidationData
from plotting.plotmodelhistory import plotHistory
from processing.getdata import numberOfClasses
from keras.src import layers
from keras.src.optimizers import Adam
from keras.src.applications.vgg19 import VGG19

#Retrieving data
trainingDataset = getTrainingData()
validationDataset = getValidationData()

#Specifying model
baseModel = VGG19(include_top = False, weights = 'imagenet', input_shape = (imageHeight, imageWidth, 3), classes=numberOfClasses)

# Freeze the pretrained weights
baseModel.trainable = False

#Build top of model
topDropout = 0.6
model = Sequential()
model.add(baseModel)
model.add(Dropout(topDropout))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64,kernel_initializer='he_uniform')) #Try with smaller and bigger
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(numberOfClasses,activation='softmax'))


#Creating callbacks
lrd = ReduceLROnPlateau(monitor = 'val_loss', patience = 20, verbose = 1, factor = 0.50, min_lr = 1e-10)
es = EarlyStopping(verbose=1, patience=20)

#Creating model
optimizer = Adam(learning_rate=1e-3) #Try with -3
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

#Train model
hist = model.fit(trainingDataset, epochs=10, validation_data=validationDataset, callbacks=[lrd,es])

#Show results
plotHistory(hist)

#Unfreeze base model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

#Start training again
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=5, validation_data=validationDataset, callbacks=[lrd,es])

#Show results
plotHistory(hist)

#Save model
model.save('./savedmodels/dog_efficientbnet0_multiclass_model.keras')