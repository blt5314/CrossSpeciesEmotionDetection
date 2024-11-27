# Importing necessary libraries
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from processing.getdata import getTrainingData
from processing.getdata import getValidationData
from plotting.plotmodelhistory import plotHistory
from config import imageWidth
from config import imageHeight
from config import numberOfClasses
from keras.src import layers
from keras.src.optimizers import Adam
from keras.src.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Name of file to save the model to
modelSaveName = "dog_vgg16_multiclass_model.keras"

# Retrieving data
trainingDataset = getTrainingData()
validationDataset = getValidationData()

# Specifying model
baseModel = VGG16(include_top=False, weights='imagenet', input_shape=(imageHeight, imageWidth, 3), classes=numberOfClasses)

# Freeze the pretrained weights
baseModel.trainable = False

# Build top of model
topDropout = 0.6
model = Sequential()
model.add(baseModel)
model.add(Dropout(topDropout))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(40, kernel_initializer='he_uniform'))  # Try with smaller and bigger
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(numberOfClasses, activation='softmax'))

# Creating callbacks
lrd = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.50, min_lr=1e-10)
es = EarlyStopping(verbose=1, patience=20)

# Creating model
optimizer = Adam(learning_rate=1e-3)  # Try with -3
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
hist = model.fit(trainingDataset, epochs=20, validation_data=validationDataset, callbacks=[lrd, es])

# Show results
plotHistory(hist)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Unfreeze base model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

# Start training again
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist = model.fit(trainingDataset, epochs=10, validation_data=validationDataset, callbacks=[lrd, es])

# Show results
plotHistory(hist)

# Plotting the training and validation loss again
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss (Fine-tuning)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save model
model.save('./savedmodels/' + modelSaveName)

# ---- New code to generate confusion matrix ----

# Step 1: Make predictions
# Get the true labels from the validation dataset
y_true = validationDataset.classes

# Predict on the validation dataset
y_pred_prob = model.predict(validationDataset)
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the class with the highest probability

# Step 2: Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Step 3: Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(numberOfClasses), yticklabels=np.arange(numberOfClasses))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
