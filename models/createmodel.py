# Import necessary libraries
import tensorflow.keras.backend as K
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from processing.getdata import getTrainingData
from processing.getdata import getValidationData
from plotting.plotmodelhistory import plotHistory
from config import imageWidth, imageHeight, numberOfClasses
from keras.src import layers
from keras.src.optimizers import Adam
from keras.src.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Custom F1-score metric
def f1_score(y_true, y_pred):
    # Convert y_true and y_pred to one-hot encoded values
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(y_pred)  # Rounds probabilities to binary values for each class

    # Calculate true positives, false positives, and false negatives
    tp = K.sum(y_true * y_pred, axis=0)  # Element-wise multiplication
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    # Compute precision, recall, and F1-score
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)  # Return the mean F1-score across all classes

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
model.add(Dense(40, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(numberOfClasses, activation='softmax'))

# Creating callbacks
lrd = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.50, min_lr=1e-10)
es = EarlyStopping(verbose=1, patience=20)

# Creating model
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score])
model.summary()

# Train model
hist = model.fit(trainingDataset, epochs=20, validation_data=validationDataset, callbacks=[lrd, es])

# Show results
plotHistory(hist)

# Unfreeze base model
for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

# Start training again
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score])
model.summary()
hist = model.fit(trainingDataset, epochs=10, validation_data=validationDataset, callbacks=[lrd, es])

# Show results
plotHistory(hist)

# Save model
model.save('./savedmodels/' + modelSaveName)

# Step 1: Make predictions
y_true = validationDataset.classes
y_pred_prob = model.predict(validationDataset)
y_pred = np.argmax(y_pred_prob, axis=1)

# Step 2: Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Step 3: Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(numberOfClasses), yticklabels=np.arange(numberOfClasses))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
