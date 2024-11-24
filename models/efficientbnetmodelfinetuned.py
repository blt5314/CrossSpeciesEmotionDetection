#Importing
from keras import Sequential, Model
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

#Try
from keras.src.applications.efficientnet_v2 import EfficientNetV2B0
from keras.src.applications.mobilenet_v3 import MobileNetV3
from keras.src.applications.resnet_v2 import ResNet50V2
from keras.src.applications.vgg16 import VGG16

#Retrieving data
trainingDataset = getTrainingData()
validationDataset = getValidationData()

#Specifying model
baseModel = EfficientNetV2B0(include_top = False,
                       weights = 'imagenet',
                       input_shape = (imageHeight, imageWidth, 3),
                       classes=numberOfClasses)#Try with pooling and no classes

# Freeze the pretrained weights
baseModel.trainable = False

#Build top of model
topDropout = 0.5
model = Sequential()
model.add(baseModel)

#Try with 32
model.add(Dropout(topDropout))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(numberOfClasses,activation='softmax'))

"""
model.add(GlobalAveragePooling2D(name="avg_pool"))
model.add(BatchNormalization())
model.add(Dropout(topDropout, name="top_dropout"))
model.add(Dense(numberOfClasses, activation="softmax", name="pred"))
"""

#Creating callbacks
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)
es = EarlyStopping(verbose=1, patience=20)

#Creating model
optimizer = Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

#Train model
hist = model.fit(trainingDataset, epochs=5, validation_data=validationDataset, callbacks=[lrd,es])

#Show results
plotHistory(hist)

#Unfreeze base model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

#Start training again
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(trainingDataset, epochs=5, validation_data=validationDataset, callbacks=[lrd,es])

#Show results
plotHistory(hist)

#Save model
model.save('./savedmodels/dog_efficientbnet0_multiclass_model.keras')