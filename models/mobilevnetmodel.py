from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt



# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs Available: ", gpus)
else:
    print("No GPUs detected.")


# Dataset Preparation
imageWidth, imageHeight = 224, 224
batchSize = 32
dataDirectory = '../data/dog'

# Load the datasets
trainingDataset = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

valDataset = image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

# Extract class names and number of classes
class_names = trainingDataset.class_names
num_classes = len(class_names)

# Normalization and Prefetching
AUTOTUNE = tf.data.AUTOTUNE
trainingDataset = trainingDataset.map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=AUTOTUNE)
valDataset = valDataset.map(lambda x, y: (x / 255.0, y)).prefetch(buffer_size=AUTOTUNE)

# Define the base model
base_model = MobileNetV2(
    input_shape=(imageHeight, imageWidth, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze the base model initially

# Build the new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
epochs = 20  # Reduce epochs for faster training
history = model.fit(
    trainingDataset,
    validation_data=valDataset,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Plot Training and Validation Accuracy/Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
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
