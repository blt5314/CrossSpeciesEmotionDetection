import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Define custom metrics: F1 score, Precision, and Recall
def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

def precision_metric(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')

def recall_metric(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')


# Function to load 100 random images from the dataset
def load_random_images(dataset_path, image_size=(224, 224), batch_size=100):
    data_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    data_flow = data_gen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    # Use `next()` to get a batch of images
    images, labels = next(data_flow)

    # Ensure only 100 samples
    if images.shape[0] > 100:
        images = images[:100]
        labels = labels[:100]
    return images, labels


# Function to test a model on a dataset
def test_model(model_path, dataset_path, image_size=(224, 224)):
    # Load the model with custom metrics (f1_score, precision, recall)
    model = load_model(model_path, custom_objects={'f1_score': f1_metric, 'precision': precision_metric, 'recall': recall_metric})
    print(f"Loaded model from {model_path}")

    # Load 100 random images and their true labels
    images, true_labels = load_random_images(dataset_path, image_size=image_size, batch_size=100)
    print(f"Loaded 100 images from {dataset_path}")

    # Get predictions
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)  # Convert one-hot encoding to class indices

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2%}")

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.2f}")

    # Calculate Precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    print(f"Precision: {precision:.2f}")

    # Calculate Recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f"Recall: {recall:.2f}")

    return accuracy, f1, precision, recall


# Main block to test multiple models
if __name__ == "__main__":
    # Specify datasets and models
    datasets = ["../data/combined", "../data/dog", "../data/human"]
    models = [
        "../models/savedmodels/combined_vgg16_multiclass_model.keras",
        "../models/savedmodels/dog_vgg16_multiclass_model.keras",
        "../models/savedmodels/human_vgg16_multiclass_model.keras"
    ]

    # Image size
    image_width, image_height = 224, 224

    # Test each model on each dataset
    for dataset_path in datasets:
        print(f"\nTesting on dataset: {dataset_path}")
        for model_path in models:
            print(f"Testing model: {model_path}")
            test_model(model_path, dataset_path, image_size=(image_width, image_height))
