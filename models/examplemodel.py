#Importing
import numpy as np
import tensorflow as tf
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from keras.src.applications.efficientnet_v2 import EfficientNetV2
from keras.src.models import Model
from keras.src.models import Sequential
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from keras.src import layers
from keras.src.losses.losses import SparseCategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

#Specifying datasets
imageWidth, imageHeight = 224, 224
batchSize = 32
dataDirectory = '../data/dog'

trainingDataset = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset = 'training',
    seed = 123,
    image_size = (imageHeight, imageWidth),
    batch_size = batchSize
)

val_ds = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.3,
    subset = 'validation',
    seed = 123,
    image_size = (imageHeight, imageWidth),
    batch_size = batchSize
)

#Normalization
num_of_classes = len(trainingDataset.class_names)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = trainingDataset.map(lambda x, y: (normalization_layer(x),y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

#Model
model = Sequential([
    layers.Rescaling(1./255.0, input_shape=(imageHeight, imageWidth, 3)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(num_of_classes)
])

model.compile(optimizer='adam', loss = SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(trainingDataset, epochs=10, validation_data=val_ds)

#Plotting accuracy
epochs=10
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

epochs=10
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

data_augmentation = Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(imageHeight, imageWidth, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images, _ in trainingDataset.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_of_classes, name="outputs")
])

model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(
  trainingDataset,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()