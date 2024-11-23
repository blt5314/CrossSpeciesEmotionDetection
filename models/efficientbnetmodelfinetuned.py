#Importing
from keras.src.applications.efficientnet import EfficientNetB0, EfficientNetB5
from processing.getdata import imageWidth
from processing.getdata import imageHeight
from processing.dataprocessing import getProcessedTrainingData
from processing.dataprocessing import getProcessedValidationData
from plotting.plotmodelhistory import plotHistory
from processing.getdata import numberOfClasses
from keras.src import layers
from keras.src import Model
from keras.src.optimizers import Adam

#Retrieving data
trainingDataset = getProcessedTrainingData()
validationDataset = getProcessedValidationData()


#Specifying inputs
inputs = layers.Input(shape=(imageHeight, imageWidth, 3))

#Specifying model
model = EfficientNetB5(
    include_top = False,
    weights = 'imagenet',
    input_tensor = inputs,
)

# Freeze the pretrained weights
model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)
top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(numberOfClasses, activation="softmax", name="pred")(x)

#Creating model
model = Model(inputs, outputs, name="EfficientNet")
optimizer = Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=10, validation_data=validationDataset)

#Show results
plotHistory(hist)

#Unfreeze model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

#Start training again
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=5, validation_data=validationDataset)

#Show results
plotHistory(hist)

#Save model
model.save('dog_efficientbnet0_finetuned_multiclass_model.keras')